"""
logging.py - Logging utilities for the PokerGTO project

This module provides functions to set up and configure logging
for different components of the PokerGTO project.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Optional


def setup_logger(name: str, level: int = logging.INFO,
                 log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with the specified name and configuration.

    Args:
        name: Name of the logger
        level: Logging level (default: INFO)
        log_file: Optional path to a log file

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler if a log file is specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_log_path(module_name: str, base_dir: Optional[Path] = None) -> Path:
    """
    Get the appropriate log file path for a module.

    Args:
        module_name: Name of the module
        base_dir: Base directory for log files (default: ./data/logs)

    Returns:
        Path object pointing to the log file
    """
    if base_dir is None:
        # Default log directory is relative to the project root
        project_root = Path(__file__).parent.parent.parent
        base_dir = project_root / "data" / "game_logs"

    # Create the directory if it doesn't exist
    base_dir.mkdir(parents=True, exist_ok=True)

    # Create a log file name based on the module name
    log_file = base_dir / f"{module_name}.log"

    return log_file


def configure_global_logging(level: int = logging.INFO,
                             log_file: Optional[str] = None,
                             console: bool = True) -> None:
    """
    Configure global logging settings.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional path to a log file
        console: Whether to log to the console (default: True)
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Add file handler if a log file is specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def log_system_info(logger: logging.Logger) -> None:
    """
    Log system information including Python version, available packages, etc.

    Args:
        logger: Logger to use for logging
    """
    import platform
    import torch
    import numpy as np

    logger.info("System Information:")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"NumPy version: {np.__version__}")

    # Log CUDA information if available
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            logger.info(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  - Total memory: {props.total_memory / 1024 ** 3:.2f} GB")
            logger.info(f"  - Compute capability: {props.major}.{props.minor}")
    else:
        logger.info("CUDA not available")

    logger.info(f"Number of CPU threads: {torch.get_num_threads()}")


def setup_tensorboard_logger(log_dir: Path) -> None:
    """
    Set up TensorBoard logging.

    Args:
        log_dir: Directory to save TensorBoard logs
    """
    try:
        from torch.utils.tensorboard import SummaryWriter

        # Create the directory if it doesn't exist
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create the TensorBoard writer
        writer = SummaryWriter(log_dir=str(log_dir))

        return writer
    except ImportError:
        logging.warning("TensorBoard not available. Install with: pip install tensorboard")
        return None