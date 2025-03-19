#!/usr/bin/env python
"""
calculate.py - GTO Strategy Calculation Using Discounted Regret Minimization

This script implements and runs the Discounted Regret Minimization algorithm
to calculate Game Theory Optimal (GTO) strategies for Texas Hold'em poker.
The resulting model is saved for later use in play.py and train.py.
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Add the project directory to the path so we can import our modules
project_root = Path(os.path.dirname(os.path.abspath(__file__)))
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.engine.game_state import GameState
from src.engine.evaluator import HandEvaluator
from src.engine.rules import PokerRules
from src.algorithms.discounted_regret_gpu import GPUDiscountedRegretMinimization
from src.utils.cuda_utils import setup_cuda, get_cuda_info
from src.utils.logging import setup_logger
from src.utils.visualization import plot_convergence, plot_strategy
import config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Calculate GTO poker strategy using Discounted Regret Minimization.')
    parser.add_argument('--iterations', type=int, default=config.DRM_PARAMS['iterations'],
                        help='Number of iterations to run the algorithm')
    parser.add_argument('--discount', type=float, default=config.DRM_PARAMS['discount_factor'],
                        help='Discount factor for regrets (0.0-1.0)')
    parser.add_argument('--save_path', type=str, default=str(config.DRM_MODEL_PATH),
                        help='Path to save the resulting model')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Path to save the log file')
    parser.add_argument('--eval_interval', type=int, default=100,
                        help='Interval to evaluate and report progress')
    parser.add_argument('--save_interval', type=int, default=config.DRM_PARAMS['save_interval'],
                        help='Interval to save the model')
    parser.add_argument('--batch_size', type=int, default=config.DRM_PARAMS['batch_size'],
                        help='Batch size for GPU computation')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--num_threads', type=int, default=4,
                        help='Number of CPU threads for parallel operations')
    return parser.parse_args()


def setup_environment(args):
    """Setup the computation environment and logging."""
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger("calculate", log_level, args.log_file)

    # Configure CUDA if available
    use_cuda = config.USE_CUDA and not args.no_cuda
    if use_cuda:
        device, cuda_info = setup_cuda(config.CUDA_DEVICE_ID)
        logger.info(f"Using CUDA: {cuda_info}")
    else:
        device = torch.device("cpu")
        logger.info(f"Using CPU: {torch.get_num_threads()} threads")
        if config.USE_CUDA and args.no_cuda:
            logger.info("CUDA is available but disabled by command line argument")
        elif not config.USE_CUDA:
            logger.info("CUDA is not available on this system")

    return logger, device


def create_model(args, device):
    """Create and initialize the DRM model."""
    # Initialize the poker game engine components
    rules = PokerRules(
        small_blind=config.GAME_PARAMS['small_blind'],
        big_blind=config.GAME_PARAMS['big_blind'],
        stack_size=config.GAME_PARAMS['stack_size'],
        max_raises=config.GAME_PARAMS['max_raises_per_street']
    )

    evaluator = HandEvaluator()

    # Initialize the GPU-accelerated DRM algorithm
    drm = GPUDiscountedRegretMinimization(
        rules=rules,
        evaluator=evaluator,
        discount_factor=args.discount,
        device=device,
        batch_size=args.batch_size,
        num_threads=args.num_threads
    )

    return drm


def train_model(drm, args, logger):
    """Run the DRM training loop."""
    logger.info(f"Starting DRM calculation for {args.iterations} iterations")
    logger.info(f"Discount factor: {args.discount}")

    # Create output directory if it doesn't exist
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Initialize tracking variables
    start_time = time.time()
    exploitability_history = []
    convergence_history = []
    best_exploitability = float('inf')

    # Progress bar
    pbar = tqdm(range(1, args.iterations + 1), desc="Training DRM model")

    # Main training loop
    for iteration in pbar:
        # Run a training iteration
        metrics = drm.iterate()
        convergence = metrics.get('convergence', 0.0)
        convergence_history.append(convergence)

        # Periodically evaluate the model
        if iteration % args.eval_interval == 0:
            elapsed = time.time() - start_time
            exploitability = drm.compute_exploitability()
            exploitability_history.append((iteration, exploitability))

            # Calculate iterations per second
            iter_per_sec = iteration / max(0.001, elapsed)

            # Calculate estimated remaining time
            remaining_iterations = args.iterations - iteration
            estimated_seconds = remaining_iterations / max(0.001, iter_per_sec)

            # Format time as HH:MM:SS
            hours, remainder = divmod(estimated_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

            # Update progress bar
            pbar.set_postfix({
                'Expl': f"{exploitability:.6f}",
                'Conv': f"{convergence:.6f}",
                'Iter/s': f"{iter_per_sec:.1f}",
                'ETA': time_str
            })

            # Log more detailed metrics
            logger.info(f"Iteration {iteration}/{args.iterations}: "
                        f"Exploitability = {exploitability:.6f}, "
                        f"Convergence = {convergence:.6f}, "
                        f"Iter/s = {iter_per_sec:.1f}, "
                        f"Info states = {metrics.get('num_info_states', 0)}, "
                        f"Cache hit rate = {metrics.get('cache_hit_rate', 0.0):.2f}")

            # Check if this is the best model so far
            if exploitability < best_exploitability:
                best_exploitability = exploitability
                drm.save_model(save_path / "best_model")
                logger.info(f"New best model saved with exploitability {best_exploitability:.6f}")

        # Periodically save the model
        if iteration % args.save_interval == 0:
            checkpoint_path = save_path / f"checkpoint_{iteration}"
            drm.save_model(checkpoint_path)

            # Also save visualization
            if iteration >= 100:  # Only save visualization after some iterations
                plot_convergence(convergence_history, save_path / f"convergence_{iteration}.png")

    # Save the final model
    drm.save_model(save_path / "final_model")
    logger.info(f"Final model saved with exploitability {best_exploitability:.6f}")

    # Save exploitability history for later analysis
    np.save(save_path / "exploitability_history.npy", np.array(exploitability_history))

    # Generate and save final visualizations
    plot_convergence(convergence_history, save_path / "convergence_final.png")

    # Report training stats
    total_time = time.time() - start_time
    iterations_per_second = args.iterations / max(0.001, total_time)
    logger.info(f"Training completed in {total_time:.2f} seconds ({iterations_per_second:.1f} iterations/sec)")
    logger.info(f"Final exploitability: {best_exploitability:.6f}")
    logger.info(f"Models saved to {save_path}")

    return drm


def main():
    """Main function to run the DRM calculation."""
    # Parse arguments
    args = parse_args()

    # Setup environment
    logger, device = setup_environment(args)

    try:
        # Create model
        logger.info("Initializing GPU-accelerated DRM model...")
        drm = create_model(args, device)

        # Train model
        trained_model = train_model(drm, args, logger)

        logger.info("DRM calculation completed successfully")
        return 0

    except Exception as e:
        logger.exception(f"Error during DRM calculation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())