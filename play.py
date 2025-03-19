#!/usr/bin/env python
"""
play.py - Interactive poker play with advanced GTO-based assistance

This script allows users to play poker with intelligent AI opponents
using Game Theory Optimal (GTO) strategies derived from machine learning.
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
import random
import torch
from pathlib import Path
import json

# Add the project directory to the path so we can import our modules
project_root = Path(os.path.dirname(os.path.abspath(__file__)))
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.engine.game_state import GameState, Street, Action
from src.engine.evaluator import HandEvaluator
from src.engine.rules import PokerRules
from src.algorithms.discounted_regret import DiscountedRegretMinimization, InfoState
from src.utils.cuda_utils import setup_cuda, get_cuda_info
from src.utils.logging import setup_logger
from src.interface.tooltips import TooltipGenerator
from src.utils.terminology import get_term_definition, get_short_definition
import config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Interactive poker play with GTO assistance.')
    parser.add_argument('--model_path', type=str, default=str(config.ADAPTIVE_MODEL_PATH),
                        help='Path to the trained model')
    parser.add_argument('--mode', type=str, choices=['interactive', 'auto', 'training'],
                        default='interactive',
                        help='Play mode: interactive, auto, or training')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Path to save the log file')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--num_hands', type=int, default=10,
                        help='Number of hands to play in non-interactive modes')
    parser.add_argument('--tooltip_detail', type=str, choices=['low', 'medium', 'high'],
                        default=config.INTERFACE.get('tooltip_detail', 'medium'),
                        help='Level of detail for tooltips')
    parser.add_argument('--show_help', action='store_true',
                        help='Show detailed poker terminology help before starting')
    return parser.parse_args()


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


def get_hand_description(evaluator, cards, community_cards=None):
    """
    Generate a descriptive string for a hand.

    Args:
        evaluator: HandEvaluator instance
        cards: Hole cards
        community_cards: Community cards (optional)

    Returns:
        Descriptive string of the hand
    """
    if community_cards is None:
        community_cards = []

    # Get hand strength
    hand_strength = evaluator.evaluate_hand(cards, community_cards)
    hand_type = evaluator.get_hand_type(hand_strength)

    # Convert card indices to readable strings
    card_strs = [print_card(card) for card in cards]
    community_card_strs = [print_card(card) for card in community_cards]

    return (f"{hand_type} " +
            f"(Hole: {' '.join(card_strs)}" +
            (f" | Community: {' '.join(community_card_strs)}" if community_cards else "") + ")")


def get_strategy_recommendation(drm, state, hole_cards, community_cards):
    """
    Get a strategy recommendation from the GTO model.

    Args:
        drm: Discounted Regret Minimization model
        state: Current game state
        hole_cards: Player's hole cards
        community_cards: Current community cards

    Returns:
        Sorted list of action recommendations
    """
    info_state = InfoState(
        player_id=state.current_player,
        hole_cards=hole_cards,
        community_cards=community_cards,
        action_history=state.action_history,
        street=state.street
    )

    # Get the GTO strategy for this state
    strategy = drm.get_average_strategy(info_state)

    # Sort actions by probability for better readability
    sorted_actions = sorted(strategy.items(), key=lambda x: x[1], reverse=True)

    return sorted_actions


def calculate_outs(hole_cards, community_cards, evaluator):
    """
    Calculate the number of outs (cards that would improve the hand).

    Args:
        hole_cards: Player's hole cards
        community_cards: Current community cards
        evaluator: Hand evaluator instance

    Returns:
        Estimated number of outs
    """
    # Current hand strength
    current_strength = evaluator.evaluate_hand(hole_cards, community_cards)

    # Check all remaining cards to see if they improve the hand
    all_cards = set(range(52))
    used_cards = set(hole_cards + community_cards)
    remaining_cards = list(all_cards - used_cards)

    outs = 0
    for card in remaining_cards:
        new_community = community_cards + [card]
        new_strength = evaluator.evaluate_hand(hole_cards, new_community)
        if new_strength > current_strength:
            outs += 1

    return outs


def calculate_pot_odds(state, player_index):
    """
    Calculate pot odds for a potential call.

    Args:
        state: Current game state
        player_index: Index of the player

    Returns:
        Tuple of (pot_odds_percentage, call_amount, pot_size)
    """
    max_bet = max(state.bets)
    call_amount = max_bet - state.bets[player_index]

    if call_amount <= 0:
        return 0, 0, state.pot

    pot_size = state.pot + sum(state.bets)
    pot_odds = call_amount / (pot_size + call_amount)
    pot_odds_percentage = pot_odds * 100

    return pot_odds_percentage, call_amount, pot_size


def determine_position(player_index, button_pos, num_players):
    """
    Determine the player's position at the table.

    Args:
        player_index: Index of the player
        button_pos: Position of the dealer button
        num_players: Total number of players

    Returns:
        String describing the position
    """
    positions = ["Button", "Small Blind", "Big Blind"]

    # In heads-up, positions are simplified
    if num_players == 2:
        if player_index == button_pos:
            return "Button/Small Blind"
        else:
            return "Big Blind"

    # Calculate position relative to button
    pos_from_button = (player_index - button_pos) % num_players

    if pos_from_button == 0:
        return "Button"
    elif pos_from_button == 1:
        return "Small Blind"
    elif pos_from_button == 2:
        return "Big Blind"
    elif pos_from_button == 3:
        return "Under The Gun (UTG)"
    elif pos_from_button >= num_players - 2:
        return "Cutoff"
    elif pos_from_button >= num_players - 3:
        return "Hijack"
    else:
        return "Middle Position"


def ai_choose_action(drm, state, hole_cards, community_cards, logger):
    """
    AI strategy for choosing an action.

    Args:
        drm: Discounted Regret Minimization model
        state: Current game state
        hole_cards: AI's hole cards
        community_cards: Current community cards
        logger: Logging instance

    Returns:
        Tuple of (chosen_action, bet_amount)
    """
    # Get strategy recommendations
    recommendations = get_strategy_recommendation(drm, state, hole_cards, community_cards)

    # Get legal actions
    legal_actions = state.get_legal_actions()

    if not legal_actions:
        return None, 0.0

    # Choose action based on strategy probabilities
    strategy_dict = dict(recommendations)
    action_probs = [strategy_dict.get(action, 0.0) for action, _, _ in legal_actions]

    # Normalize probabilities
    total_prob = sum(action_probs)
    if total_prob == 0:
        # Fallback to uniform distribution
        action_probs = [1.0 / len(legal_actions)] * len(legal_actions)
    else:
        action_probs = [prob / total_prob for prob in action_probs]

    # Choose action probabilistically
    chosen_action_idx = np.random.choice(len(legal_actions), p=action_probs)
    chosen_action, min_amount, max_amount = legal_actions[chosen_action_idx]

    # Determine bet amount if action is bet or raise
    bet_amount = 0.0
    if chosen_action in [Action.BET, Action.RAISE]:
        # Intelligent bet sizing based on strategy and hand strength
        # Here we add some randomness to make it more realistic
        bet_range = max_amount - min_amount
        bet_amount = min_amount + random.random() * bet_range * 0.5

    logger.info(f"AI Strategy: {[(a.name, f'{p:.2f}') for a, p in recommendations]}")
    logger.info(f"AI Chose: {chosen_action.name} " +
                (f"Amount: {bet_amount:.2f}" if bet_amount > 0 else ""))

    return chosen_action, bet_amount


def print_hand_summary(state, evaluator):
    """
    Print a summary of the hand, including hand strengths.

    Args:
        state: Final game state
        evaluator: HandEvaluator instance
    """
    print("\n--- Hand Summary ---")
    for i in range(state.num_players):
        if not state.folded[i]:
            hand_desc = get_hand_description(
                evaluator,
                state.player_cards[i],
                state.community_cards
            )
            print(f"Player {i}: {hand_desc}")

    if len(set(not folded for folded in state.folded)) == 1:
        print("Winner: By Fold")
    else:
        strengths = [
            evaluator.evaluate_hand(state.player_cards[i], state.community_cards)
            for i in range(state.num_players)
        ]
        winners = [
            i for i, strength in enumerate(strengths)
            if strength == max(strengths) and not state.folded[i]
        ]

        if len(winners) == 1:
            print(f"Winner: Player {winners[0]}")
        else:
            print(f"Split Pot: Players {winners}")
    print("-------------------\n")


def show_terminology_help():
    """Display helpful poker terminology information to the user."""
    print("\n========== POKER TERMINOLOGY GUIDE ==========")
    print("\nBasic Poker Actions:")
    print("  FOLD   - " + get_short_definition("fold"))
    print("  CHECK  - " + get_short_definition("check"))
    print("  CALL   - " + get_short_definition("call"))
    print("  BET    - Putting chips into the pot when no one else has bet")
    print("  RAISE  - " + get_short_definition("raise"))
    print("  ALL-IN - " + get_short_definition("all_in"))

    print("\nPoker Rounds (Streets):")
    print("  PREFLOP - Betting round before any community cards are dealt")
    print("  FLOP    - " + get_short_definition("flop"))
    print("  TURN    - " + get_short_definition("turn"))
    print("  RIVER   - " + get_short_definition("river"))

    print("\nHand Rankings (Strongest to Weakest):")
    print("  1. Royal Flush     - A, K, Q, J, 10 of the same suit")
    print("  2. Straight Flush  - Five consecutive cards of the same suit")
    print("  3. Four of a Kind  - Four cards of the same rank")
    print("  4. Full House      - Three cards of one rank and two of another")
    print("  5. Flush           - Five cards of the same suit")
    print("  6. Straight        - Five consecutive cards of any suit")
    print("  7. Three of a Kind - Three cards of the same rank")
    print("  8. Two Pair        - Two different pairs")
    print("  9. Pair            - Two cards of the same rank")
    print("  10. High Card      - Highest card when no other hand is made")

    print("\nKey Concepts:")
    print("  POSITION    - Where you act relative to other players (earlier is worse)")
    print("  POT ODDS    - The ratio of the current pot size to the cost of a call")
    print("  OUTS        - " + get_short_definition("outs"))
    print("  GTO         - Game Theory Optimal play, a perfectly balanced strategy")

    print("\nType 'help' during the game for a random poker tip.")
    print("==============================================\n")

    input("Press Enter to continue...")


def play_interactive(drm, rules, evaluator, logger, tooltip_detail="medium"):
    """Run an interactive poker session with GTO assistance."""
    # Initialize tooltip generator
    tooltips = TooltipGenerator(detail_level=tooltip_detail)

    print("\nWelcome to PokerGTO Interactive Play!")
    print("You'll play heads-up Texas Hold'em with GTO-based recommendations.")
    print("Type 'help' during the game for poker tips, or 'term [concept]' for explanations.")

    playing = True
    player_wins = 0
    ai_wins = 0

    while playing:
        # Initialize game state
        state = GameState(
            num_players=2,
            stack_size=config.GAME_PARAMS['stack_size'],
            small_blind=config.GAME_PARAMS['small_blind'],
            big_blind=config.GAME_PARAMS['big_blind']
        )

        # Deal hole cards
        hole_cards = rules.deal_hole_cards(num_players=2)
        state.deal_hole_cards(hole_cards)

        # Play the hand
        hand_finished = False
        while not hand_finished:
            if state.is_terminal():
                # Evaluate the hand if it went to showdown
                if not state.folded[0] and not state.folded[1]:
                    if len(state.community_cards) < 5:
                        # Deal remaining community cards if needed
                        remaining = 5 - len(state.community_cards)
                        remaining_cards = rules.get_random_cards(remaining)
                        state.community_cards.extend(remaining_cards)

                print_hand_summary(state, evaluator)

                # Determine winner
                player_hand = evaluator.evaluate_hand(
                    state.player_cards[0], state.community_cards)
                ai_hand = evaluator.evaluate_hand(
                    state.player_cards[1], state.community_cards)

                if player_hand > ai_hand:
                    player_wins += 1
                    print("Congratulations! You won the hand!")
                elif player_hand < ai_hand:
                    ai_wins += 1
                    print("AI won the hand!")
                else:
                    print("It's a tie!")

                hand_finished = True
                continue

            # Current player's turn
            current_player = state.current_player

            if current_player == 0:  # Human player
                # Display current game state information
                print("\n" + "=" * 50)
                street_name = state.street.name
                print(f"\nStreet: {street_name} - {tooltips.get_street_tooltip(state.street)}")

                # Show player position
                position = determine_position(0, state.button_pos, state.num_players)
                print(f"Your position: {position}")

                # Display pot size and stack information
                print(f"Pot: {state.pot:.1f} - Your stack: {state.stacks[0]:.1f} - AI stack: {state.stacks[1]:.1f}")

                # Display cards
                player_cards = state.player_cards[0]
                print(f"Your cards: {print_cards(player_cards)}")

                if state.community_cards:
                    print(f"Community cards: {print_cards(state.community_cards)}")

                    # Calculate hand strength and outs
                    hand_desc = get_hand_description(evaluator, player_cards, state.community_cards)
                    print(f"Your hand: {hand_desc}")

                    # Calculate outs
                    outs = calculate_outs(player_cards, state.community_cards, evaluator)
                    if outs > 0:
                        print(f"Outs: {outs} - {tooltips.get_outs_explanation(outs)}")

                # Get pot odds if applicable
                pot_odds, call_amount, pot_size = calculate_pot_odds(state, 0)
                if pot_odds > 0:
                    print(tooltips.get_pot_odds_advice(call_amount, pot_size))

                # Get GTO recommendations
                recommendations = get_strategy_recommendation(
                    drm, state, player_cards, state.community_cards
                )

                # Display recommendations
                print("\nGTO Recommendations:")
                for action, prob in recommendations:
                    print(f"  {action.name}: {prob:.1%} - {tooltips.get_action_tooltip(action)}")

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
                    choice_input = input("\nEnter your choice (number), 'help', or 'term [concept]': ").strip()

                    # Handle help request
                    if choice_input.lower() == 'help':
                        print("\nTip: " + tooltips.get_help_tip())
                        continue

                    # Handle terminology lookup
                    if choice_input.lower().startswith('term '):
                        term = choice_input[5:].strip()
                        explanation = tooltips.get_concept_explanation(term)
                        print(f"\n{term.title()}: {explanation}")
                        continue

                    try:
                        choice = int(choice_input) - 1
                        if 0 <= choice < len(legal_actions):
                            valid_choice = True
                        else:
                            print(f"Invalid choice. Please enter a number between 1 and {len(legal_actions)}.")
                    except ValueError:
                        print("Invalid input. Please enter a number, 'help', or 'term [concept]'.")

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

            else:  # AI player
                print("\nAI is thinking...")
                time.sleep(0.5)  # Add a small delay to make it feel more natural

                # Get AI's action
                ai_action, ai_amount = ai_choose_action(
                    drm,
                    state,
                    state.player_cards[1],
                    state.community_cards,
                    logger
                )

                if ai_action is not None:
                    print(f"AI {ai_action.name}s" + (f" {ai_amount:.1f}" if ai_amount > 0 else ""))
                    street_complete = state.apply_action(ai_action, ai_amount)

            # Deal new street if necessary
            if street_complete and not state.is_terminal():
                if state.street == Street.PREFLOP:
                    # Deal flop
                    flop = rules.deal_flop()
                    state.deal_community_cards(flop, Street.FLOP)
                    print(f"\nFLOP: {print_cards(flop)}")
                elif state.street == Street.FLOP:
                    # Deal turn
                    turn = rules.deal_turn()
                    state.deal_community_cards(turn, Street.TURN)
                    print(f"\nTURN: {print_cards(turn)}")
                elif state.street == Street.TURN:
                    # Deal river
                    river = rules.deal_river()
                    state.deal_community_cards(river, Street.RIVER)
                    print(f"\nRIVER: {print_cards(river)}")

        # Print running score
        print(f"\nScore - You: {player_wins}, AI: {ai_wins}")

        # Ask if the user wants to play another hand
        play_again = input("Play another hand? (y/n): ").lower()
        if play_again != 'y':
            playing = False

    print("\nThanks for playing!")
    print(f"Final Score - You: {player_wins}, AI: {ai_wins}")


def main():
    """Main function to run the interactive play."""
    # Parse arguments
    args = parse_args()

    # Setup environment
    logger = setup_logger("play", logging.DEBUG if args.debug else logging.INFO, args.log_file)

    # Show help if requested
    if args.show_help:
        show_terminology_help()

    # Setup CUDA
    use_cuda = config.USE_CUDA and not args.no_cuda
    if use_cuda:
        device, cuda_info = setup_cuda(config.CUDA_DEVICE_ID)
        logger.info(f"Using CUDA: {cuda_info}")
    else:
        device = torch.device("cpu")
        logger.info(f"Using CPU: {torch.get_num_threads()} threads")

    try:
        # Load model
        logger.info("Loading trained model...")

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

        # Run play modes
        if args.mode == 'interactive':
            play_interactive(drm, rules, evaluator, logger, args.tooltip_detail)
        elif args.mode == 'auto':
            # Placeholder for future auto play mode implementation
            logger.info("Auto play mode not yet implemented")
        elif args.mode == 'training':
            # Placeholder for future training mode implementation
            logger.info("Training mode not yet implemented")

        logger.info("Play session completed successfully")
        return 0

    except Exception as e:
        logger.exception(f"Error during play: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())