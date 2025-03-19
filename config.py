"""
PokerGTO Collaborative Learning Configuration

This configuration supports advanced multi-model learning strategies.
"""

import os
from pathlib import Path

# Project Directory Structure
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = DATA_DIR / "logs"
TRAINING_HISTORY_DIR = DATA_DIR / "training_history"

# Create directories if they don't exist
for directory in [MODELS_DIR, DATA_DIR, LOG_DIR, TRAINING_HISTORY_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Collaborative Learning Parameters
COLLABORATIVE_LEARNING = {
    'num_models': 6,
    'games_per_model': 10000,
    'knowledge_transfer_frequency': 100,
    'transfer_rate': 0.5,
    'similarity_threshold': 0.7
}

# Game Parameters
GAME_PARAMS = {
    'stack_size': 200,                 # Initial stack size in big blinds
    'small_blind': 0.5,                # Small blind size
    'big_blind': 1.0,                  # Big blind size
    'num_players': 2,                  # Number of players (heads-up)
    'max_raises_per_street': 4,        # Maximum number of raises per street
}

# Discounted Regret Minimization Parameters
DRM_PARAMS = {
    'iterations': 1000000,             # Number of iterations to run
    'discount_factor': {
        'min': 0.90,                   # Minimum discount factor
        'max': 0.99,                   # Maximum discount factor
        'default': 0.95                # Default discount factor
    },
    'learning_rate': {
        'min': 0.001,                  # Minimum learning rate
        'max': 0.1,                    # Maximum learning rate
        'default': 0.01                # Default learning rate
    },
    'exploration_factor': {
        'min': 0.1,                    # Minimum exploration
        'max': 0.5,                    # Maximum exploration
        'default': 0.2                 # Default exploration
    }
}

# Model Paths
DRM_MODEL_PATH = MODELS_DIR / "drm_model"
ADAPTIVE_MODEL_PATH = MODELS_DIR / "adaptive_model"
COLLABORATIVE_MODEL_PATH = MODELS_DIR / "collaborative_models"

# Slumbot API Configuration
SLUMBOT_API = {
    'base_url': 'http://www.slumbot.com/api/v1/',
    'endpoints': {
        'new_hand': 'new_hand',
        'act': 'act',
    },
    'retries': 3,                      # Number of API call retries
    'timeout': 5,                      # Timeout in seconds
    'verify_ssl': True,                # Verify SSL certificates
}

# Training Parameters
TRAINING_PARAMS = {
    'num_games': 10000,                # Number of games to play
    'save_interval': 100,              # Save model every N games
    'eval_interval': 500,              # Evaluate model every N games
    'log_interval': 10,                # Log metrics every N games
    'checkpoint_interval': 1000,       # Create model checkpoint every N games
    'improvement_threshold': 0.01,     # Minimum improvement to save new model
}

# Logging Configuration
LOGGING = {
    'level': 'INFO',                   # Logging level
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': LOG_DIR / 'poker_gto.log', # Log file path
    'console': True,                   # Output logs to console
}

# Performance Tracking
PERFORMANCE_TRACKING = {
    'metrics': [
        'win_rate',
        'exploitability',
        'collaborative_score'
    ],
    'window_size': 100,                # Number of recent games to track
    'moving_average_window': 10        # Window for computing moving averages
}

# Advanced Machine Learning Parameters
ML_ADVANCED_PARAMS = {
    'knowledge_transfer': {
        'enabled': True,
        'transfer_rate': 0.5,
        'similarity_threshold': 0.7
    },
    'ensemble_learning': {
        'enabled': True,
        'ensemble_method': 'weighted_average'
    },
    'meta_learning': {
        'enabled': True,
        'adaptation_rate': 0.1
    }
}

# System Performance Configuration
SYSTEM_PERFORMANCE = {
    'max_cpu_threads': 8,              # Maximum CPU threads to use
    'gpu_acceleration': True,          # Enable GPU acceleration
    'memory_limit_mb': 4096,           # Memory limit for model training
}