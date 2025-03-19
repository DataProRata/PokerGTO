"""
visualization.py - Visualization utilities for the PokerGTO project

This module provides functions to visualize poker strategies, convergence,
and other aspects of the GTO calculation and training process.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Import for type annotations only
from src.algorithms.discounted_regret import DiscountedRegretMinimization


def setup_plotting_style():
    """Set up the plotting style for consistent visualizations."""
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12


def plot_convergence(convergence_history: List[float], save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot the convergence of the DRM algorithm over iterations.

    Args:
        convergence_history: List of convergence metrics from each iteration
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure object
    """
    # Set up the plotting style
    setup_plotting_style()

    # Create the figure
    fig, ax = plt.subplots()

    # Plot the convergence
    iterations = range(1, len(convergence_history) + 1)
    ax.plot(iterations, convergence_history, 'b-', linewidth=2)

    # Add a trendline
    if len(convergence_history) > 1:
        z = np.polyfit(iterations, np.log(np.array(convergence_history) + 1e-10), 1)
        p = np.poly1d(z)
        ax.plot(iterations, np.exp(p(iterations)), 'r--', linewidth=1, alpha=0.7,
                label=f'Trend: $e^{{{z[0]:.4f}x + {z[1]:.4f}}}$')

    # Set the scale to logarithmic if values vary by orders of magnitude
    if max(convergence_history) / (min(convergence_history) + 1e-10) > 100:
        ax.set_yscale('log')

    # Add labels and title
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Convergence Metric')
    ax.set_title('DRM Algorithm Convergence')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Improve layout
    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_exploitability(exploitability_history: List[Tuple[int, float]],
                       save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot the exploitability of the strategy over iterations.

    Args:
        exploitability_history: List of (iteration, exploitability) tuples
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure object
    """
    # Set up the plotting style
    setup_plotting_style()

    # Create the figure
    fig, ax = plt.subplots()

    # Unpack the exploitability history
    iterations, exploitability = zip(*exploitability_history)

    # Plot the exploitability
    ax.plot(iterations, exploitability, 'g-', linewidth=2)

    # Add a trendline
    if len(iterations) > 1:
        z = np.polyfit(np.log(iterations), np.log(exploitability), 1)
        p_log = np.poly1d(z)
        ax.plot(iterations, np.exp(p_log(np.log(iterations))), 'r--', linewidth=1, alpha=0.7,
                label=f'Trend: $e^{{{z[0]:.2f}\\ln(x) + {z[1]:.2f}}}$')

    # Set the scales to logarithmic
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Add labels and title
    ax.set_xlabel('Iterations (log scale)')
    ax.set_ylabel('Exploitability (log scale)')
    ax.set_title('Strategy Exploitability vs. Iterations')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Improve layout
    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_strategy(drm: 'DiscountedRegretMinimization',
                 save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot a visualization of the current poker strategy.

    Args:
        drm: Discounted Regret Minimization instance
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure object
    """
    # Set up the plotting style
    setup_plotting_style()

    # This is a simplified placeholder visualization
    # In a real implementation, you would visualize actual strategy profiles
    # for various key situations (e.g., different hand strengths, betting rounds)

    # Create a figure with 2x2 subplots for different situations
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Placeholder data - in a real implementation, we would extract this from drm
    situations = [
        "Strong Hand - Preflop",
        "Medium Hand - Flop",
        "Weak Hand - Turn",
        "Bluffing - River"
    ]

    actions = ["Fold", "Check/Call", "Bet/Raise"]

    # Generate some example strategy profiles
    strategies = [
        [0.05, 0.25, 0.70],  # Strong hand strategy
        [0.15, 0.60, 0.25],  # Medium hand strategy
        [0.40, 0.50, 0.10],  # Weak hand strategy
        [0.30, 0.20, 0.50]   # Bluffing strategy
    ]

    # Plot each strategy
    for i, (ax, situation, strategy) in enumerate(zip(axs.flat, situations, strategies)):
        bars = ax.bar(actions, strategy, color=['red', 'blue', 'green'])

        # Add percentage labels on the bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0%}',
                    ha='center', va='bottom')

        ax.set_ylim(0, 1)
        ax.set_ylabel('Probability')
        ax.set_title(situation)

    # Add an overall title
    plt.suptitle(f'Poker Strategy Visualization (After {drm.iteration_count} iterations)', fontsize=16)

    # Improve layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_training_progress(train_history: Dict[str, List[float]],
                          save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot the training progress from the training history.

    Args:
        train_history: Dictionary with training metrics
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure object
    """
    # Set up the plotting style
    setup_plotting_style()

    # Create a figure with multiple subplots
    n_metrics = len(train_history)
    fig, axs = plt.subplots(n_metrics, 1, figsize=(10, 3 * n_metrics), sharex=True)

    # If there's only one metric, axs is not a list
    if n_metrics == 1:
        axs = [axs]

    # Plot each metric
    for i, (metric_name, values) in enumerate(train_history.items()):
        axs[i].plot(range(1, len(values) + 1), values)
        axs[i].set_ylabel(metric_name)
        axs[i].grid(True, alpha=0.3)

    # Add x-axis label to the bottom subplot
    axs[-1].set_xlabel('Episodes')

    # Add an overall title
    plt.suptitle('Training Progress', fontsize=16)

    # Improve layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def create_card_visualization(card_indices: List[int],
                             save_path: Optional[Path] = None) -> plt.Figure:
    """
    Create a visual representation of poker cards.

    Args:
        card_indices: List of card indices (0-51)
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure object
    """
    # Set up the plotting style
    setup_plotting_style()

    # Map card indices to ranks and suits
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suits = ['♥', '♦', '♣', '♠']
    suit_colors = ['red', 'red', 'black', 'black']

    # Create a figure
    n_cards = len(card_indices)
    fig, axs = plt.subplots(1, n_cards, figsize=(n_cards * 2, 3))

    # If there's only one card, axs is not a list
    if n_cards == 1:
        axs = [axs]

    # Plot each card
    for i, card_idx in enumerate(card_indices):
        rank_idx = card_idx % 13
        suit_idx = card_idx // 13

        rank = ranks[rank_idx]
        suit = suits[suit_idx]
        color = suit_colors[suit_idx]

        # Create a white rectangle with a black border for the card
        axs[i].add_patch(plt.Rectangle((0, 0), 1, 1.5, facecolor='white', edgecolor='black'))

        # Add the rank and suit
        axs[i].text(0.5, 0.75, rank + suit, fontsize=24, ha='center', va='center', color=color)

        # Remove axes
        axs[i].axis('off')

        # Set the limits
        axs[i].set_xlim(0, 1)
        axs[i].set_ylim(0, 1.5)

    # Improve layout
    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_hand_strength_distribution(hand_strengths: List[float],
                                  save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot the distribution of hand strengths.

    Args:
        hand_strengths: List of hand strength values
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure object
    """
    # Set up the plotting style
    setup_plotting_style()

    # Create the figure
    fig, ax = plt.subplots()

    # Plot the histogram
    ax.hist(hand_strengths, bins=30, color='blue', alpha=0.7)

    # Add a kernel density estimate
    if len(hand_strengths) > 1:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(hand_strengths)
        x = np.linspace(min(hand_strengths), max(hand_strengths), 1000)
        ax.plot(x, kde(x) * len(hand_strengths) / 30, 'r-', linewidth=2)

    # Add labels and title
    ax.set_xlabel('Hand Strength')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Hand Strengths')
    ax.grid(True, alpha=0.3)

    # Improve layout
    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig