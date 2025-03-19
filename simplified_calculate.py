#!/usr/bin/env python
"""
simplified_calculate_fixed.py - GTO Strategy Calculation with OpenMP fix

This script includes a fix for OpenMP library conflicts and continues the
pragmatic approach to DRM with reliable progress display and GPU usage.
"""

# Fix for OpenMP library conflicts - MUST be at the top of the file
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import time
import argparse
import logging
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Add the project directory to the path
project_root = Path(os.path.dirname(os.path.abspath(__file__)))
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.engine.game_state import GameState
from src.engine.evaluator import HandEvaluator
from src.engine.rules import PokerRules
from src.algorithms.simplified_drm import SimplifiedDRM
from src.utils.cuda_utils import setup_cuda, get_cuda_info
from src.utils.logging import setup_logger
from src.utils.visualization import plot_convergence
import config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Calculate GTO poker strategy with reliable progress.')
    parser.add_argument('--iterations', type=int, default=10000,
                        help='Number of iterations to run the algorithm')
    parser.add_argument('--discount', type=float, default=0.95,
                        help='Discount factor for regrets (0.0-1.0)')
    parser.add_argument('--save_path', type=str, default=str(config.DRM_MODEL_PATH),
                        help='Path to save the resulting model')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size for GPU computation')
    parser.add_argument('--eval_interval', type=int, default=100,
                        help='Interval to evaluate and report progress')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='Interval to save the model')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--num_threads', type=int, default=24,
                        help='Number of CPU threads for parallel operations')
    return parser.parse_args()


def setup_environment(args):
    """Setup the computation environment and logging."""
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger("simplified_calculate", log_level)

    # Configure CUDA if available
    use_cuda = config.USE_CUDA and not args.no_cuda
    if use_cuda:
        device, cuda_info = setup_cuda(config.CUDA_DEVICE_ID)
        logger.info(f"Using CUDA: {cuda_info}")
    else:
        device = torch.device("cpu")
        logger.info(f"Using CPU: {torch.get_num_threads()} threads")

    # Set number of threads for CPU operations
    torch.set_num_threads(args.num_threads)
    logger.info(f"Set PyTorch threads to {args.num_threads}")

    # Print OpenMP status
    logger.info(f"OpenMP duplicate lib setting: {os.environ.get('KMP_DUPLICATE_LIB_OK', 'FALSE')}")

    return logger, device


def create_model(args, device, logger):
    """Create and initialize the simplified DRM model."""
    # Initialize poker game engine components
    rules = PokerRules(
        small_blind=config.GAME_PARAMS['small_blind'],
        big_blind=config.GAME_PARAMS['big_blind'],
        stack_size=config.GAME_PARAMS['stack_size'],
        max_raises=config.GAME_PARAMS['max_raises_per_street']
    )

    evaluator = HandEvaluator()

    # Initialize the simplified DRM algorithm
    drm = SimplifiedDRM(
        rules=rules,
        evaluator=evaluator,
        discount_factor=args.discount,
        device=device,
        batch_size=args.batch_size,
        debug=args.debug
    )

    return drm


def train_model(drm, args, logger):
    """Run the DRM training loop with reliable progress display."""
    logger.info(f"Starting simplified DRM calculation for {args.iterations} iterations")
    logger.info(f"Discount factor: {args.discount}")

    # Create output directory if it doesn't exist
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Initialize tracking variables
    start_time = time.time()
    iteration_times = []
    convergence_history = []

    # Main training loop with progress bar
    pbar = tqdm(range(1, args.iterations + 1), desc="Training DRM model")

    try:
        for iteration in pbar:
            iter_start = time.time()

            # Run one iteration
            metrics = drm.iterate()
            convergence = metrics.get('convergence', 0.0)
            convergence_history.append(convergence)

            # Record iteration time
            iter_time = time.time() - iter_start
            iteration_times.append(iter_time)
            avg_iter_time = sum(iteration_times[-100:]) / min(len(iteration_times), 100)

            # Update progress bar with useful metrics
            states_processed = metrics.get('num_states_processed', 0)
            states_total = metrics.get('num_info_states', 0)
            iter_per_sec = 1.0 / max(0.001, avg_iter_time)

            # Calculate ETA
            remaining_iterations = args.iterations - iteration
            eta_seconds = remaining_iterations * avg_iter_time
            hours, remainder = divmod(eta_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            eta_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

            # Update progress bar
            pbar.set_postfix({
                'Conv': f"{convergence:.4f}",
                'States': states_total,
                'it/s': f"{iter_per_sec:.1f}",
                'ETA': eta_str
            })

            # Periodically log more detailed metrics
            if iteration % args.eval_interval == 0:
                elapsed = time.time() - start_time
                logger.info(f"Iteration {iteration}/{args.iterations}: "
                            f"States = {states_total}, "
                            f"Convergence = {convergence:.6f}, "
                            f"Time/iter = {avg_iter_time:.3f}s, "
                            f"GPU memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f}MB")

            # Periodically save the model
            if iteration % args.save_interval == 0:
                checkpoint_path = save_path / f"checkpoint_{iteration}"
                drm.save_model(checkpoint_path)
                logger.info(f"Checkpoint saved at iteration {iteration}")

                # Save convergence plot
                if len(convergence_history) > 1:
                    plot_convergence(convergence_history, save_path / f"convergence_{iteration}.png")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    finally:
        # Save the final model
        drm.save_model(save_path / "final_model")

        # Generate and save final visualization
        if len(convergence_history) > 1:
            plot_convergence(convergence_history, save_path / "convergence_final.png")

        # Report training stats
        total_time = time.time() - start_time
        states_per_sec = metrics.get('num_info_states', 0) / max(0.001, total_time)
        logger.info(f"Training completed or interrupted after {drm.iteration_count} iterations")
        logger.info(f"Total time: {total_time:.2f} seconds ({states_per_sec:.1f} states/sec)")
        logger.info(f"Final model saved to {save_path}")

    return drm


def main():
    """Main function to run the simplified DRM calculation."""
    # Parse arguments
    args = parse_args()

    # Setup environment
    logger, device = setup_environment(args)

    try:
        # Create model
        logger.info("Initializing simplified DRM model...")
        drm = create_model(args, device, logger)

        # Train model with reliable progress display
        trained_model = train_model(drm, args, logger)

        logger.info("DRM calculation completed successfully")
        return 0

    except Exception as e:
        logger.exception(f"Error during DRM calculation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())