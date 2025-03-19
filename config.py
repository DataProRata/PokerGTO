"""
PokerGTO Configuration Settings

This module contains all configuration parameters for the PokerGTO project,
including algorithm parameters, file paths, and hardware settings.
"""

import os
import torch
from pathlib import Path

# Project directory structure
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = DATA_DIR / "game_logs"
TRAINING_HISTORY_DIR = DATA_DIR / "training_history"

# Create directories if they don't exist
for directory in [MODELS_DIR, DATA_DIR, LOG_DIR, TRAINING_HISTORY_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Default model paths
DRM_MODEL_PATH = MODELS_DIR / "drm_model"
ADAPTIVE_MODEL_PATH = MODELS_DIR / "adaptive_model"

# Hardware Settings
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
NUM_GPU = torch.cuda.device_count() if USE_CUDA else 0
CUDA_DEVICE_ID = 0  # Default to first GPU
TORCH_THREADS = 4  # Number of CPU threads to use when GPU is not available

# Algorithm Parameters
# Discounted Regret Minimization Parameters
DRM_PARAMS = {
    "iterations": 1000000,             # Number of iterations to run
    "discount_factor": 0.95,           # Regret discount factor (0.0-1.0)
    "exploration_factor": 0.6,         # Initial exploration factor
    "exploration_decay": 0.9999,       # Decay rate for exploration
    "min_exploration": 0.05,           # Minimum exploration probability
    "batch_size": 1024,                # Training batch size
    "learning_rate": 0.001,            # Learning rate for neural models
    "update_target_every": 100,        # Update target network every N iterations
    "memory_size": 100000,             # Experience replay buffer size
    "warm_start_size": 1000,           # Samples before learning starts
    "save_interval": 10000,            # Save model every N iterations
}

# Game parameters
GAME_PARAMS = {
    "stack_size": 200,                 # Initial stack size in big blinds
    "small_blind": 0.5,                # Small blind size
    "big_blind": 1.0,                  # Big blind size
    "num_players": 2,                  # Number of players (heads-up only for now)
    "max_raises_per_street": 4,        # Maximum number of raises per street
}

# Card deck and evaluation
NUM_RANKS = 13                         # Ace through King
NUM_SUITS = 4                          # Hearts, Diamonds, Clubs, Spades
RANK_TO_STRING = {
    0: '2', 1: '3', 2: '4', 3: '5', 4: '6', 5: '7', 6: '8',
    7: '9', 8: 'T', 9: 'J', 10: 'Q', 11: 'K', 12: 'A'
}
SUIT_TO_STRING = {0: 'h', 1: 'd', 2: 'c', 3: 's'}

# Slumbot API settings
SLUMBOT_API = {
    "base_url": "http://www.slumbot.com/api/v1/",
    "endpoints": {
        "new_hand": "new_hand",
        "act": "act",
    },
    "retries": 3,                      # Number of API call retries
    "timeout": 5,                      # Timeout in seconds
    "verify_ssl": True,                # Verify SSL certificates
}

# Training parameters
TRAINING_PARAMS = {
    "num_games": 10000,                # Number of games to play
    "save_interval": 100,              # Save model every N games
    "eval_interval": 500,              # Evaluate model every N games
    "log_interval": 10,                # Log metrics every N games
    "checkpoint_interval": 1000,       # Create model checkpoint every N games
    "improvement_threshold": 0.01,     # Minimum improvement to save new model
}

# Logging settings
LOGGING = {
    "level": "INFO",                   # Logging level (DEBUG, INFO, WARNING, ERROR)
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOG_DIR / "poker_gto.log", # Log file path
    "console": True,                   # Output logs to console
}

# Play interface settings
INTERFACE = {
    "tooltips": True,                  # Enable tooltips
    "tooltip_detail": "high",          # Detail level (low, medium, high)
    "animation_speed": 0.5,            # Speed of animations (seconds)
    "default_mode": "interactive",     # Default play mode (interactive, auto)
}