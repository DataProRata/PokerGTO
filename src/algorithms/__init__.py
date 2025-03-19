"""
Algorithms module for the PokerGTO project.

This module contains the core algorithms for GTO strategy calculation:
- DiscountedRegretMinimization: DRM implementation for poker
- Other algorithms may be added in the future
"""

from src.algorithms.discounted_regret import DiscountedRegretMinimization

__all__ = ['DiscountedRegretMinimization']