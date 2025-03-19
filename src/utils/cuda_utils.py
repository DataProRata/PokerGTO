"""
cuda_utils.py - Utility functions for CUDA setup and management

This module provides functions to setup and manage CUDA devices,
check system capabilities, and optimize CUDA performance.
"""

import torch
import logging
from typing import Tuple, Dict, Optional


def get_cuda_info() -> Dict[str, any]:
    """
    Get information about available CUDA devices.

    Returns:
        Dictionary with CUDA information
    """
    info = {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": []
    }

    if info["available"]:
        # Get current device
        info["current_device"] = torch.cuda.current_device()

        # Get info for each device
        for i in range(info["device_count"]):
            device_info = {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "capability": torch.cuda.get_device_capability(i),
                "total_memory": torch.cuda.get_device_properties(i).total_memory / (1024 ** 3),  # in GB
            }
            info["devices"].append(device_info)

    return info


def setup_cuda(device_id: Optional[int] = None) -> Tuple[torch.device, Dict[str, any]]:
    """
    Setup CUDA for computation.

    Args:
        device_id: Specific CUDA device ID to use, or None for auto-selection

    Returns:
        Tuple of (torch.device, device_info)
    """
    if not torch.cuda.is_available():
        return torch.device("cpu"), {"available": False}

    cuda_info = get_cuda_info()

    # Validate device_id
    if device_id is not None:
        if device_id >= cuda_info["device_count"]:
            raise ValueError(
                f"Requested CUDA device {device_id} but only {cuda_info['device_count']} devices are available"
            )
        selected_device = device_id
    else:
        # Auto-select the device with the most memory
        if cuda_info["device_count"] > 1:
            max_memory = 0
            selected_device = 0

            for i, device in enumerate(cuda_info["devices"]):
                if device["total_memory"] > max_memory:
                    max_memory = device["total_memory"]
                    selected_device = i
        else:
            selected_device = 0

    # Set the device
    torch.cuda.set_device(selected_device)
    device = torch.device(f"cuda:{selected_device}")

    # Get selected device info
    selected_info = cuda_info["devices"][selected_device]

    # Set some CUDA optimization flags
    torch.backends.cudnn.benchmark = True  # May improve performance for fixed-size inputs

    # Check for TF32 support (available on Ampere and later GPUs)
    if selected_info["capability"][0] >= 8:  # Ampere has compute capability 8.x
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    return device, selected_info


def optimize_cuda_memory(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """
    Apply CUDA memory optimizations to a PyTorch model.

    Args:
        model: PyTorch model to optimize
        device: CUDA device to use

    Returns:
        Optimized model
    """
    # Move model to device
    model = model.to(device)

    # Enable automatic mixed precision (AMP) if available
    if hasattr(torch.cuda, "amp") and torch.cuda.is_available():
        model = torch.cuda.amp.autocast()(model)

    return model


def get_optimal_threads(cuda_available: bool = None) -> int:
    """
    Determine the optimal number of threads for CPU computation.

    Args:
        cuda_available: Whether CUDA is available, or None to auto-detect

    Returns:
        Optimal number of CPU threads
    """
    if cuda_available is None:
        cuda_available = torch.cuda.is_available()

    # If CUDA is available, we'll use fewer CPU threads
    if cuda_available:
        return min(4, torch.get_num_threads())
    else:
        # Use all available CPU threads
        return torch.get_num_threads()


def benchmark_device(device: torch.device, matrix_size: int = 2000) -> float:
    """
    Benchmark the computation speed of a device.

    Args:
        device: Device to benchmark
        matrix_size: Size of test matrices

    Returns:
        Operations per second (higher is better)
    """
    import time

    # Create test matrices
    a = torch.randn(matrix_size, matrix_size, device=device)
    b = torch.randn(matrix_size, matrix_size, device=device)

    # Warm up
    for _ in range(5):
        _ = torch.matmul(a, b)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    start_time = time.time()
    iterations = 10

    for _ in range(iterations):
        c = torch.matmul(a, b)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()

    # Calculate operations per second (approximate for matrix multiply)
    ops = 2 * (matrix_size ** 3) * iterations  # Approximate FLOPs for matrix multiply
    seconds = end_time - start_time
    ops_per_second = ops / seconds

    return ops_per_second