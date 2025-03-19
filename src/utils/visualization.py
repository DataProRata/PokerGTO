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


def plot_action_frequencies(action_history: Dict[str, Dict[str, int]],
                           save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot the frequency of different actions across different streets.

    Args:
        action_history: Dictionary mapping streets to action frequency dictionaries
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure object
    """
    # Set up the plotting style
    setup_plotting_style()

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Extract streets and actions
    streets = list(action_history.keys())
    all_actions = set()
    for street_actions in action_history.values():
        all_actions.update(street_actions.keys())
    all_actions = sorted(all_actions)

    # Set up bar positions
    bar_width = 0.8 / len(all_actions)
    positions = np.arange(len(streets))

    # Plot bars for each action
    for i, action in enumerate(all_actions):
        action_counts = [action_history[street].get(action, 0) for street in streets]

        # Calculate percentages
        street_totals = [sum(action_history[street].values()) for street in streets]
        percentages = [count / total * 100 if total > 0 else 0
                      for count, total in zip(action_counts, street_totals)]

        # Plot the bars
        offset = bar_width * (i - len(all_actions) / 2 + 0.5)
        bars = ax.bar(positions + offset, percentages, bar_width,
                     label=action, alpha=0.7)

        # Add percentage labels
        for bar, percentage in zip(bars, percentages):
            if percentage > 5:  # Only show labels for significant percentages
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{percentage:.1f}%', ha='center', va='bottom', fontsize=8)

    # Customize the plot
    ax.set_xticks(positions)
    ax.set_xticklabels(streets)
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Action Frequencies by Street')
    ax.legend(title='Actions')
    ax.grid(True, alpha=0.3, axis='y')

    # Set y-axis to a reasonable range
    ax.set_ylim(0, 100)

    # Improve layout
    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_showdown_equity(equity_by_hand: Dict[str, float],
                        save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot the showdown equity for different hand types.

    Args:
        equity_by_hand: Dictionary mapping hand types to equity percentages
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure object
    """
    # Set up the plotting style
    setup_plotting_style()

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Sort hand types by equity
    hand_types = sorted(equity_by_hand.items(), key=lambda x: x[1], reverse=True)
    hand_names, equity_values = zip(*hand_types)

    # Plot horizontal bars
    bars = ax.barh(hand_names, equity_values, color='green', alpha=0.7)

    # Add percentage labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2,
               f'{width:.1f}%', va='center')

    # Customize the plot
    ax.set_xlabel('Equity (%)')
    ax.set_title('Showdown Equity by Hand Type')
    ax.grid(True, alpha=0.3, axis='x')

    # Set x-axis to a reasonable range
    ax.set_xlim(0, max(equity_values) * 1.1)

    # Improve layout
    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_strategy_heatmap(street: str, hand_strengths: List[float],
                         actions: List[str], probabilities: List[List[float]],
                         save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot a heatmap of action probabilities for different hand strengths.

    Args:
        street: The street (e.g., "Flop", "Turn", "River")
        hand_strengths: List of hand strength values
        actions: List of action names
        probabilities: 2D list of action probabilities for each hand strength
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure object
    """
    # Set up the plotting style
    setup_plotting_style()

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create the heatmap
    im = ax.imshow(probabilities, cmap='viridis')

    # Add a colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Probability', rotation=-90, va="bottom")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(actions)))
    ax.set_yticks(np.arange(len(hand_strengths)))
    ax.set_xticklabels(actions)

    # Format hand strengths for readable y-axis labels
    strength_labels = [f"{strength:.2f}" for strength in hand_strengths]
    ax.set_yticklabels(strength_labels)

    # Rotate the x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add labels and title
    ax.set_xlabel('Action')
    ax.set_ylabel('Hand Strength')
    ax.set_title(f'Strategy Heatmap - {street}')

    # Add text annotations showing the probability values
    for i in range(len(hand_strengths)):
        for j in range(len(actions)):
            text = ax.text(j, i, f"{probabilities[i][j]:.2f}",
                          ha="center", va="center", color="white" if probabilities[i][j] < 0.7 else "black")

    # Improve layout
    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def create_strategy_dashboard(drm: 'DiscountedRegretMinimization',
                             save_path: Optional[Path] = None) -> plt.Figure:
    """
    Create a comprehensive dashboard visualizing the poker strategy.

    Args:
        drm: Discounted Regret Minimization instance
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure object
    """
    # Set up the plotting style
    setup_plotting_style()

    # Create a figure with a grid of subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3)

    # Subplot for convergence (top left)
    ax_conv = fig.add_subplot(gs[0, 0])
    # Placeholder data - would extract from drm in real implementation
    iterations = range(1, 101)
    conv_values = [1/i for i in iterations]
    ax_conv.plot(iterations, conv_values, 'b-')
    ax_conv.set_xlabel('Iterations')
    ax_conv.set_ylabel('Convergence')
    ax_conv.set_title('Algorithm Convergence')
    ax_conv.set_yscale('log')

    # Subplot for exploitability (top middle)
    ax_expl = fig.add_subplot(gs[0, 1])
    # Placeholder data
    expl_values = [2/i for i in iterations]
    ax_expl.plot(iterations, expl_values, 'g-')
    ax_expl.set_xlabel('Iterations')
    ax_expl.set_ylabel('Exploitability')
    ax_expl.set_title('Strategy Exploitability')
    ax_expl.set_yscale('log')

    # Subplot for overall strategy distribution (top right)
    ax_dist = fig.add_subplot(gs[0, 2])
    # Placeholder data
    actions = ['Fold', 'Check/Call', 'Bet/Raise']
    action_freqs = [0.2, 0.5, 0.3]
    ax_dist.bar(actions, action_freqs, color=['red', 'blue', 'green'])
    ax_dist.set_ylabel('Frequency')
    ax_dist.set_title('Overall Action Distribution')

    # Subplots for different streets (middle row)
    streets = ['Preflop', 'Flop', 'Turn', 'River']
    street_actions = [
        [0.1, 0.4, 0.5],  # Preflop
        [0.2, 0.5, 0.3],  # Flop
        [0.3, 0.4, 0.3],  # Turn
        [0.4, 0.3, 0.3]   # River
    ]

    for i, (street, freqs) in enumerate(zip(streets, street_actions)):
        ax_street = fig.add_subplot(gs[1, i % 3] if i < 3 else gs[2, 0])
        ax_street.bar(actions, freqs, color=['red', 'blue', 'green'])
        ax_street.set_title(f'{street} Strategy')
        if i >= 1:  # Skip y-label for first subplot
            ax_street.set_ylabel('Probability')

    # Heatmap for hand strength vs. action (bottom middle)
    ax_heat = fig.add_subplot(gs[2, 1])
    # Placeholder data
    hand_ranges = ['0-25%', '25-50%', '50-75%', '75-100%']
    heat_data = [
        [0.7, 0.2, 0.1],  # Weakest hands
        [0.3, 0.5, 0.2],  # Weak-medium
        [0.1, 0.4, 0.5],  # Medium-strong
        [0.0, 0.3, 0.7]   # Strongest hands
    ]
    im = ax_heat.imshow(heat_data, cmap='viridis')
    ax_heat.set_xticks(np.arange(len(actions)))
    ax_heat.set_yticks(np.arange(len(hand_ranges)))
    ax_heat.set_xticklabels(actions)
    ax_heat.set_yticklabels(hand_ranges)
    ax_heat.set_title('Strategy by Hand Strength')

    # Win rate chart (bottom right)
    ax_win = fig.add_subplot(gs[2, 2])
    # Placeholder data
    hand_types = ['High Card', 'Pair', 'Two Pair', 'Trips', 'Straight', 'Flush', 'Full House', 'Quads']
    win_rates = [0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.95, 0.99]
    # Truncate to fit
    ax_win.barh(hand_types[-4:], win_rates[-4:], color='green', alpha=0.7)
    ax_win.set_xlabel('Win Rate')
    ax_win.set_title('Win Rate by Hand Type')

    # Add an overall title
    plt.suptitle(f'Poker Strategy Dashboard (After {drm.iteration_count} iterations)',
                fontsize=20, fontweight='bold', y=0.98)

    # Improve layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the title

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig