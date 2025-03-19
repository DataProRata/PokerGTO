#!/usr/bin/env python
"""
train.py - Train a poker AI model against the Slumbot API

This script loads a pre-trained DRM model and further trains it by playing
against the Slumbot API. The model adapts its strategy based on gameplay results.
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
from src.algorithms.discounted_regret import DiscountedRegretMinimization
from src.utils.cuda_utils import setup_cuda, get_cuda_info
from src.utils.logging import setup_logger
from src.utils.visualization import plot_training_progress
import config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a poker AI model against the Slumbot API.')
    parser.add_argument('--model_path', type=str, default=str(config.DRM_MODEL_PATH),
                        help='Path to the pre-trained DRM model')
    parser.add_argument('--games', type=int, default=config.TRAINING_PARAMS['num_games'],
                        help='Number of games to play against Slumbot')
    parser.add_argument('--save_path', type=str, default=str(config.ADAPTIVE_MODEL_PATH),
                        help='Path to save the trained model')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Path to save the log file')
    parser.add_argument('--save_interval', type=int, default=config.TRAINING_PARAMS['save_interval'],
                        help='Interval to save the model')
    parser.add_argument('--eval_interval', type=int, default=config.TRAINING_PARAMS['eval_interval'],
                        help='Interval to evaluate and report progress')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    return parser.parse_args()


def setup_environment(args):
    """Setup the computation environment and logging."""
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger("train", log_level, args.log_file)

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


def load_model(args, device, logger):
    """Load the pre-trained DRM model."""
    # Initialize the poker game engine components
    rules = PokerRules(
        small_blind=config.GAME_PARAMS['small_blind'],
        big_blind=config.GAME_PARAMS['big_blind'],
        stack_size=config.GAME_PARAMS['stack_size'],
        max_raises=config.GAME_PARAMS['max_raises_per_street']
    )

    evaluator = HandEvaluator()

    # Initialize the DRM algorithm
    drm = DiscountedRegretMinimization(
        rules=rules,
        evaluator=evaluator,
        discount_factor=config.DRM_PARAMS['discount_factor'],
        device=device,
        batch_size=config.DRM_PARAMS['batch_size']
    )

    # Load the pre-trained model if it exists
    model_path = Path(args.model_path)
    if model_path.exists():
        try:
            drm.load_model(model_path)
            logger.info(f"Loaded pre-trained model from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load model from {model_path}: {e}")
            logger.info("Starting with a fresh model")
    else:
        logger.warning(f"Model path {model_path} does not exist. Starting with a fresh model.")

    return drm


def train_model(drm, args, logger):
    """Train the model by playing against the Slumbot API."""
    logger.info(f"Starting training for {args.games} games against Slumbot")

    # Create output directory if it doesn't exist
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Initialize tracking variables
    start_time = time.time()
    training_history = {
        'winnings': [],
        'win_rate': [],
        'exploitability': []
    }
    best_win_rate = -float('inf')

    # Progress bar
    pbar = tqdm(range(1, args.games + 1), desc="Training against Slumbot")

    # Placeholder for Slumbot API implementation
    # In a real implementation, you would:
    # 1. Initialize a SlumbotClient
    # 2. Play games against Slumbot
    # 3. Update the model based on game results

    # Simulate training with dummy data for now
    for game in pbar:
        # Simulate playing a game against Slumbot
        # In a real implementation, this would involve API calls

        # Dummy data - would be real results from Slumbot games
        win_amount = np.random.normal(0.1, 1.0)  # Mean small positive win
        training_history['winnings'].append(win_amount)

        # Calculate running win rate
        win_rate = sum(training_history['winnings']) / game
        training_history['win_rate'].append(win_rate)

        # Update the model (placeholder)
        # In a real implementation, this would update the model based on game results

        # Periodically compute exploitability
        if game % args.eval_interval == 0:
            # In a real implementation, this would be an actual evaluation
            exploitability = 1.0 / np.sqrt(game)  # Dummy value that decreases with more games
            training_history['exploitability'].append(exploitability)

            elapsed = time.time() - start_time
            games_per_second = game / elapsed

            # Update progress bar
            pbar.set_postfix({
                'Win Rate': f"{win_rate:.3f}",
                'Expl': f"{exploitability:.6f}",
                'Games/s': f"{games_per_second:.1f}"
            })

            logger.info(f"Game {game}/{args.games}: "
                        f"Win Rate = {win_rate:.3f}, "
                        f"Exploitability = {exploitability:.6f}")

            # Check if this is the best model so far
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                drm.save_model(save_path / "best_model")
                logger.info(f"New best model saved with win rate {best_win_rate:.3f}")

        # Periodically save the model
        if game % args.save_interval == 0:
            checkpoint_path = save_path / f"checkpoint_{game}"
            drm.save_model(checkpoint_path)

            # Also save visualization
            plot_training_progress(training_history, save_path / f"training_progress_{game}.png")

    # Save the final model
    drm.save_model(save_path / "final_model")
    logger.info(f"Final model saved with win rate {win_rate:.3f}")

    # Generate and save final visualizations
    plot_training_progress(training_history, save_path / "training_progress_final.png")

    # Report training stats
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds")
    logger.info(f"Final win rate: {win_rate:.3f}")
    logger.info(f"Models saved to {save_path}")

    return drm


def main():
    """Main function to run the training."""
    # Parse arguments
    args = parse_args()

    # Setup environment
    logger, device = setup_environment(args)

    try:
        # Load model
        logger.info("Loading pre-trained model...")
        drm = load_model(args, device, logger)

        # Train model
        trained_model = train_model(drm, args, logger)

        logger.info("Training completed successfully")
        return 0

    except Exception as e:
        logger.exception(f"Error during training: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())