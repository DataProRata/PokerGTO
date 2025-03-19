#!/usr/bin/env python
"""
play.py - Interactive poker play with GTO-based assistance

This script allows users to play poker with assistance from the trained
GTO models. It provides real-time strategy recommendations and tooltips.
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
import torch
from pathlib import Path
import random

# Add the project directory to the path so we can import our modules
project_root = Path(os.path.dirname(os.path.abspath(__file__)))
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.engine.game_state import GameState, Street, Action
from src.engine.evaluator import HandEvaluator
from src.engine.rules import PokerRules
from src.algorithms.discounted_regret import DiscountedRegretMinimization
from src.utils.cuda_utils import setup_cuda, get_cuda_info
from src.utils.logging import setup_logger
from src.utils.visualization import create_card_visualization
import config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Interactive poker play with GTO assistance.')
    parser.add_argument('--model_path', type=str, default=str(config.ADAPTIVE_MODEL_PATH),
                        help='Path to the trained model')
    parser.add_argument('--mode', type=str, choices=['interactive', 'auto', 'tooltips'],
                        default=config.INTERFACE['default_mode'],
                        help='Play mode: interactive, auto, or tooltips')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Path to save the log file')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    return parser.parse_args()


def setup_environment(args):
    """Setup the computation environment and logging."""
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger("play", log_level, args.log_file)

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
    """Load the trained poker model."""
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

    # Load the trained model if it exists
    model_path = Path(args.model_path)
    if model_path.exists():
        try:
            drm.load_model(model_path)
            logger.info(f"Loaded trained model from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load model from {model_path}: {e}")
            logger.info("Starting with a fresh model")
    else:
        logger.warning(f"Model path {model_path} does not exist. Starting with a fresh model.")

    return drm, rules, evaluator


def print_card(card_idx):
    """Convert a card index to a readable string."""
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suits = ['♥', '♦', '♣', '♠']

    rank_idx = card_idx % 13
    suit_idx = card_idx // 13

    return f"{ranks[rank_idx]}{suits[suit_idx]}"


def print_cards(cards):
    """Print a list of cards in a readable format."""
    return ' '.join(print_card(card) for card in cards)


def print_game_state(state, player_perspective=0):
    """Print the current game state in a readable format."""
    print("\n" + "=" * 60)
    print(f"Street: {state.street.name}")
    print(f"Pot: {state.pot:.1f}")

    # Print community cards if any
    if state.community_cards:
        print(f"Community cards: {print_cards(state.community_cards)}")
    else:
        print("Community cards: None")

    # Print player information
    for i in range(state.num_players):
        # Only show the current player's hole cards
        cards_str = print_cards(state.player_cards[i]) if i == player_perspective else "XX XX"
        position = "Button" if i == state.button_pos else "Big Blind" if i == (
                    state.button_pos + 2) % state.num_players else "Small Blind"

        status = "Active"
        if state.folded[i]:
            status = "Folded"
        elif state.all_in[i]:
            status = "All-in"

        print(
            f"Player {i} ({position}): {cards_str} | Stack: {state.stacks[i]:.1f} | Bet: {state.bets[i]:.1f} | {status}")

    # Print current player
    print(f"Current player: {state.current_player}")
    print("=" * 60)


def get_strategy_recommendation(drm, state, hole_cards, community_cards):
    """Get a strategy recommendation from the GTO model."""
    # This is a simplified placeholder
    # In a real implementation, you would use the DRM model to get actionable advice

    # Create an info state from the current game state
    from src.algorithms.discounted_regret import InfoState

    info_state = InfoState(
        player_id=state.current_player,
        hole_cards=hole_cards,
        community_cards=community_cards,
        action_history=state.action_history
    )

    # Get the GTO strategy for this state
    strategy = drm.get_average_strategy(info_state)

    # Sort actions by probability for better readability
    sorted_actions = sorted(strategy.items(), key=lambda x: x[1], reverse=True)

    return sorted_actions


def play_interactive(drm, rules, evaluator, logger):
    """Run an interactive poker session with GTO assistance."""
    print("\nWelcome to PokerGTO Interactive Play!")
    print("You'll play heads-up Texas Hold'em with GTO-based recommendations.")

    # Initialize game state
    state = GameState(
        num_players=2,
        stack_size=config.GAME_PARAMS['stack_size'],
        small_blind=config.GAME_PARAMS['small_blind'],
        big_blind=config.GAME_PARAMS['big_blind']
    )

    playing = True
    while playing:
        # Start a new hand
        state.reset()

        # Deal hole cards
        hole_cards = rules.deal_hole_cards(num_players=2)
        state.deal_hole_cards(hole_cards)

        # Print initial state
        print_game_state(state, player_perspective=0)

        # Play the hand
        hand_finished = False
        while not hand_finished:
            # Check if the hand is terminal
            if state.is_terminal():
                # Evaluate the hand if it went to showdown
                if not state.folded[0] and not state.folded[1]:
                    if len(state.community_cards) < 5:
                        # Deal remaining community cards if needed
                        remaining = 5 - len(state.community_cards)
                        remaining_cards = rules.deal_cards(remaining)
                        state.community_cards.extend(remaining_cards)

                    print_game_state(state, player_perspective=0)

                    # Evaluate the hands
                    strength0 = evaluator.evaluate_hand(state.player_cards[0], state.community_cards)
                    strength1 = evaluator.evaluate_hand(state.player_cards[1], state.community_cards)

                    hand_type0 = evaluator.get_hand_type(strength0)
                    hand_type1 = evaluator.get_hand_type(strength1)

                    print(f"Player 0: {hand_type0}")
                    print(f"Player 1: {hand_type1}")

                    if strength0 > strength1:
                        print("Player 0 wins!")
                    elif strength1 > strength0:
                        print("Player 1 wins!")
                    else:
                        print("It's a tie!")
                else:
                    # Someone folded
                    winner = 0 if state.folded[1] else 1
                    print(f"Player {winner} wins (opponent folded)!")

                hand_finished = True
                continue

            # Get current player
            player = state.current_player

            # If it's user's turn (player 0)
            if player == 0:
                # Get GTO recommendations
                recommendations = get_strategy_recommendation(
                    drm, state, state.player_cards[0], state.community_cards
                )

                # Display recommendations
                print("\nGTO Recommendations:")
                for action, prob in recommendations:
                    print(f"  {action.name}: {prob:.1%}")

                # Get legal actions
                legal_actions = state.get_legal_actions()

                # Display options
                print("\nLegal Actions:")
                for i, (action, min_amount, max_amount) in enumerate(legal_actions):
                    if action in [Action.BET, Action.RAISE]:
                        print(f"{i + 1}. {action.name} (min: {min_amount:.1f}, max: {max_amount:.1f})")
                    else:
                        print(f"{i + 1}. {action.name}")

                # Get user's choice
                valid_choice = False
                while not valid_choice:
                    try:
                        choice = int(input("\nEnter your choice (number): ")) - 1
                        if 0 <= choice < len(legal_actions):
                            valid_choice = True
                        else:
                            print(f"Invalid choice. Please enter a number between 1 and {len(legal_actions)}.")
                    except ValueError:
                        print("Invalid input. Please enter a number.")

                chosen_action, min_amount, max_amount = legal_actions[choice]

                # If bet or raise, get the amount
                amount = 0.0
                if chosen_action in [Action.BET, Action.RAISE]:
                    valid_amount = False
                    while not valid_amount:
                        try:
                            amount = float(input(f"Enter amount ({min_amount:.1f}-{max_amount:.1f}): "))
                            if min_amount <= amount <= max_amount:
                                valid_amount = True
                            else:
                                print(
                                    f"Invalid amount. Please enter a value between {min_amount:.1f} and {max_amount:.1f}.")
                        except ValueError:
                            print("Invalid input. Please enter a number.")

                # Apply the action
                street_complete = state.apply_action(chosen_action, amount)

                # Deal new street if necessary
                if street_complete and not state.is_terminal():
                    if state.street == Street.PREFLOP:
                        # Deal flop
                        flop = rules.deal_flop()
                        state.deal_community_cards(flop, Street.FLOP)
                    elif state.street == Street.FLOP:
                        # Deal turn
                        turn = rules.deal_turn()
                        state.deal_community_cards(turn, Street.TURN)
                    elif state.street == Street.TURN:
                        # Deal river
                        river = rules.deal_river()
                        state.deal_community_cards(river, Street.RIVER)

                # Print updated state
                print_game_state(state, player_perspective=0)

            # If it's AI's turn (player 1)
            else:
                # Get AI's recommendation
                recommendations = get_strategy_recommendation(
                    drm, state, state.player_cards[1], state.community_cards
                )

                # Choose the action with highest probability
                chosen_action, _ = recommendations[0]

                # Get legal actions
                legal_actions = state.get_legal_actions()

                # Find the chosen action from legal actions
                action_entry = None
                for action, min_amount, max_amount in legal_actions:
                    if action == chosen_action:
                        action_entry = (action, min_amount, max_amount)
                        break

                if action_entry:
                    action, min_amount, max_amount = action_entry

                    # For bet or raise, choose a random amount between min and max
                    amount = 0.0
                    if action in [Action.BET, Action.RAISE]:
                        # In a real implementation, the model would provide guidance on sizing
                        amount = min_amount + random.random() * (max_amount - min_amount)

                    print(f"\nPlayer 1 chooses: {action.name}" +
                          (f" {amount:.1f}" if action in [Action.BET, Action.RAISE] else ""))

                    # Apply the action
                    street_complete = state.apply_action(action, amount)

                    # Deal new street if necessary
                    if street_complete and not state.is_terminal():
                        if state.street == Street.PREFLOP:
                            # Deal flop
                            flop = rules.deal_flop()
                            state.deal_community_cards(flop, Street.FLOP)
                        elif state.street == Street.FLOP:
                            # Deal turn
                            turn = rules.deal_turn()
                            state.deal_community_cards(turn, Street.TURN)
                        elif state.street == Street.TURN:
                            # Deal river
                            river = rules.deal_river()
                            state.deal_community_cards(river, Street.RIVER)

                    # Print updated state
                    print_game_state(state, player_perspective=0)
                else:
                    print("Error: AI couldn't find a legal action. Skipping turn.")
                    street_complete = True

        # Ask if the user wants to play another hand
        play_again = input("\nPlay another hand? (y/n): ")
        if play_again.lower() != 'y':
            playing = False

    print("\nThanks for playing!")


def main():
    """Main function to run the interactive play."""
    # Parse arguments
    args = parse_args()

    # Setup environment
    logger, device = setup_environment(args)

    try:
        # Load model
        logger.info("Loading trained model...")
        drm, rules, evaluator = load_model(args, device, logger)

        # Run interactive play
        if args.mode == 'interactive':
            play_interactive(drm, rules, evaluator, logger)
        else:
            logger.info(f"Mode '{args.mode}' not implemented yet")
            print(f"Mode '{args.mode}' is not implemented yet. Please use 'interactive' mode.")

        logger.info("Play session completed successfully")
        return 0

    except Exception as e:
        logger.exception(f"Error during play: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())