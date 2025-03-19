"""
discounted_regret.py - Discounted Regret Minimization implementation for poker

This module implements the Discounted Regret Minimization algorithm for calculating
GTO strategies in poker. This is an extension of Counterfactual Regret Minimization
that uses a discount factor on historical regrets to accelerate convergence.
"""

import os
import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Set, Any, Deque
from pathlib import Path
import pickle
import time
from collections import deque

from src.engine.game_state import GameState, Street, Action
from src.engine.evaluator import HandEvaluator
from src.engine.rules import PokerRules


class InfoState:
    """
    Represents an information state in a poker game.

    An information state is the perspective of a player at a particular point
    in the game, including their hole cards, visible community cards, and action history.
    """

    def __init__(self, player_id: int, hole_cards: List[int],
                community_cards: List[int], action_history: List[Tuple], street: Street):
        """
        Initialize an information state.

        Args:
            player_id: ID of the player viewing this state
            hole_cards: The player's hole cards
            community_cards: Visible community cards
            action_history: History of actions that led to this state
            street: Current street (round) of the game
        """
        self.player_id = player_id
        self.hole_cards = sorted(hole_cards) if hole_cards else []
        self.community_cards = sorted(community_cards) if community_cards else []
        self.action_history = action_history[:] if action_history else []
        self.street = street

    def __hash__(self) -> int:
        """Generate a hash for the information state."""
        # Create a hashable representation of the state
        hole_cards_tuple = tuple(self.hole_cards)
        community_cards_tuple = tuple(self.community_cards)

        # Make sure action history is hashable
        action_history_tuple = tuple(
            (p, a.value if isinstance(a, Action) else a, float(amt))
            for p, a, amt in self.action_history
        )

        return hash((self.player_id, hole_cards_tuple, community_cards_tuple,
                    action_history_tuple, self.street.value))

    def __eq__(self, other) -> bool:
        """Check if two information states are equal."""
        if not isinstance(other, InfoState):
            return False

        return (self.player_id == other.player_id and
                self.hole_cards == other.hole_cards and
                self.community_cards == other.community_cards and
                self.action_history == other.action_history and
                self.street == other.street)

    def __str__(self) -> str:
        """Return a string representation of the information state."""
        hole_str = ','.join(str(card) for card in self.hole_cards)
        comm_str = ','.join(str(card) for card in self.community_cards)
        action_str = '; '.join(
            f"Player {p}: {a.name if isinstance(a, Action) else a} {amt}"
            for p, a, amt in self.action_history
        )

        return (f"Player {self.player_id} | Street: {self.street.name} | "
                f"Hole: [{hole_str}] | Community: [{comm_str}] | Actions: [{action_str}]")


class DiscountedRegretMinimization:
    """
    Implementation of Discounted Regret Minimization for poker.

    This algorithm calculates approximate Nash equilibrium strategies
    by tracking and minimizing counterfactual regret with a discount factor
    applied to historical regrets.
    """

    def __init__(self, rules: PokerRules, evaluator: HandEvaluator,
                discount_factor: float = 0.95, device: torch.device = None,
                batch_size: int = 1024):
        """
        Initialize the DRM algorithm.

        Args:
            rules: Poker rules instance
            evaluator: Hand evaluator instance
            discount_factor: Discount factor for historical regrets (0.0-1.0)
            device: PyTorch device to use for computations
            batch_size: Batch size for parallel operations
        """
        self.logger = logging.getLogger(__name__)
        self.rules = rules
        self.evaluator = evaluator
        self.discount_factor = discount_factor
        self.device = device if device is not None else torch.device("cpu")
        self.batch_size = batch_size

        # Initialize data structures for regret tracking
        self.cumulative_regrets = {}  # Maps info state to regrets for each action
        self.strategy_sums = {}       # Maps info state to cumulative strategy profile
        self.iteration_count = 0

        self.logger.info(f"Initialized DRM with discount factor {discount_factor}")
        self.logger.info(f"Using device: {self.device}")

    def get_strategy(self, info_state: InfoState) -> Dict[Action, float]:
        """
        Get the current strategy for an information state.

        Args:
            info_state: Current information state

        Returns:
            Dictionary mapping actions to probabilities
        """
        info_state_hash = hash(info_state)

        # If we haven't seen this state before, initialize it
        if info_state_hash not in self.cumulative_regrets:
            # Create a game state to get legal actions
            game_state = self._info_state_to_game_state(info_state)
            legal_actions = game_state.get_legal_actions()

            # Initialize regrets to 0 for all legal actions
            self.cumulative_regrets[info_state_hash] = {
                action: 0.0 for action, _, _ in legal_actions
            }

            # Initialize strategy sums to 0 for all legal actions
            self.strategy_sums[info_state_hash] = {
                action: 0.0 for action, _, _ in legal_actions
            }

        # Get the cumulative regrets for this info state
        regrets = self.cumulative_regrets[info_state_hash]

        # Compute the strategy using regret matching
        strategy = self._regret_matching(regrets)

        return strategy

    def _regret_matching(self, regrets: Dict[Action, float]) -> Dict[Action, float]:
        """
        Convert regrets to a strategy using regret matching.

        Args:
            regrets: Dictionary mapping actions to regret values

        Returns:
            Dictionary mapping actions to strategy probabilities
        """
        # Strategy is proportional to positive regrets
        positive_regrets = {action: max(0.0, regret) for action, regret in regrets.items()}
        total_positive_regret = sum(positive_regrets.values())

        if total_positive_regret > 0:
            # Normalize positive regrets to form a probability distribution
            strategy = {action: regret / total_positive_regret
                      for action, regret in positive_regrets.items()}
        else:
            # If all regrets are non-positive, use a uniform strategy
            strategy = {action: 1.0 / len(regrets) for action in regrets}

        return strategy

    def _info_state_to_game_state(self, info_state: InfoState) -> GameState:
        """
        Convert an information state to a game state for simulation.

        This is a simplified version for the prototype. In a full implementation,
        this would reconstruct the complete game state from the information state.

        Args:
            info_state: Information state to convert

        Returns:
            GameState object representing the current state of the game
        """
        # Create a new game state
        game_state = GameState(
            num_players=2,  # Assuming heads-up poker for simplicity
            stack_size=self.rules.stack_size,
            small_blind=self.rules.small_blind,
            big_blind=self.rules.big_blind
        )

        # Set hole cards - for other players, use empty lists
        player_cards = [[] for _ in range(game_state.num_players)]
        player_cards[info_state.player_id] = info_state.hole_cards.copy()
        game_state.player_cards = player_cards

        # Set community cards
        game_state.community_cards = info_state.community_cards.copy()

        # Set the current street
        game_state.street = info_state.street

        # Set the current player
        game_state.current_player = info_state.player_id

        # Replay action history if provided
        if info_state.action_history:
            # This would normally update stacks, pot, etc. based on action history
            # For simplicity in the prototype, we'll just set who's to act
            game_state.action_history = info_state.action_history.copy()

        return game_state

    def update_strategy_sum(self, info_state: InfoState, strategy: Dict[Action, float],
                           reach_prob: float) -> None:
        """
        Update the cumulative strategy sum for an information state.

        Args:
            info_state: Current information state
            strategy: Current strategy for this info state
            reach_prob: Reach probability for this info state
        """
        info_state_hash = hash(info_state)

        # Initialize if not already present
        if info_state_hash not in self.strategy_sums:
            game_state = self._info_state_to_game_state(info_state)
            legal_actions = game_state.get_legal_actions()
            self.strategy_sums[info_state_hash] = {
                action: 0.0 for action, _, _ in legal_actions
            }

        # Update strategy sums weighted by reach probability
        for action, prob in strategy.items():
            if action in self.strategy_sums[info_state_hash]:
                self.strategy_sums[info_state_hash][action] += reach_prob * prob

    def get_average_strategy(self, info_state: InfoState) -> Dict[Action, float]:
        """
        Get the average strategy for an information state.

        The average strategy represents the Nash equilibrium strategy.

        Args:
            info_state: Information state to get strategy for

        Returns:
            Dictionary mapping actions to probabilities
        """
        info_state_hash = hash(info_state)

        if info_state_hash not in self.strategy_sums:
            # If we've never seen this state, return a uniform strategy
            game_state = self._info_state_to_game_state(info_state)
            legal_actions = game_state.get_legal_actions()
            return {action: 1.0 / len(legal_actions) for action, _, _ in legal_actions}

        # Get the strategy sums for this info state
        strategy_sum = self.strategy_sums[info_state_hash]
        total_sum = sum(strategy_sum.values())

        if total_sum > 0:
            # Normalize strategy sums to get the average strategy
            avg_strategy = {action: sum_val / total_sum
                          for action, sum_val in strategy_sum.items()}
        else:
            # If all sums are 0, use a uniform strategy
            avg_strategy = {action: 1.0 / len(strategy_sum)
                          for action in strategy_sum}

        return avg_strategy

    def cfr_traverse(self, hole_cards: List[List[int]], initial_state: GameState) -> float:
        """
        Non-recursive CFR tree traversal using an explicit stack.

        Args:
            hole_cards: List of hole cards for each player
            initial_state: Initial game state

        Returns:
            Expected value for player 0
        """
        # This class stores all the information we need for each state in the traversal
        class TraversalState:
            def __init__(self, game_state, reach_probs, player, return_value=None):
                self.game_state = game_state
                self.reach_probs = reach_probs
                self.player = player
                self.info_state = None
                self.strategy = None
                self.action_values = {}
                self.state_value = 0.0
                self.return_value = return_value
                self.pending_actions = []
                self.processed = False

        # Set hole cards in the initial state
        for i, cards in enumerate(hole_cards):
            initial_state.player_cards[i] = cards.copy()

        # Initialize the traversal stack with the initial state
        stack = deque([TraversalState(initial_state, [1.0, 1.0], initial_state.current_player)])

        # Process states until the stack is empty
        while stack:
            # Get the current state from the top of the stack
            current = stack[-1]

            # If this state is fully processed, pop it and update its parent
            if current.processed:
                stack.pop()

                # If the stack is now empty, we're done
                if not stack:
                    return current.state_value

                # Update the parent's action values with this state's value
                parent = stack[-1]
                action = parent.pending_actions.pop()
                parent.action_values[action] = current.state_value

                # If the parent has no more pending actions, calculate its value
                if not parent.pending_actions:
                    parent.processed = True

                    # Calculate the expected value of the parent state
                    parent.state_value = sum(parent.strategy.get(action, 0.0) * value
                                           for action, value in parent.action_values.items())

                    # Update regrets for the parent state
                    info_state_hash = hash(parent.info_state)
                    for action in parent.action_values:
                        regret = parent.action_values[action] - parent.state_value
                        cf_regret = regret * parent.reach_probs[1 - parent.player]

                        # Apply discount factor
                        if self.iteration_count > 0:
                            self.cumulative_regrets[info_state_hash][action] *= self.discount_factor

                        # Update cumulative regret
                        self.cumulative_regrets[info_state_hash][action] += cf_regret

                continue

            # Handle terminal states
            if current.game_state.is_terminal():
                # For terminal states, evaluate the hand strength
                if all(len(cards) > 0 for cards in current.game_state.player_cards):
                    # Calculate hand strengths and payoffs
                    hand_strengths = [
                        self.evaluator.evaluate_hand(cards, current.game_state.community_cards)
                        for cards in current.game_state.player_cards
                    ]
                    payoffs = current.game_state.get_payoffs(hand_strengths)
                    current.state_value = payoffs[0]  # Return value for player 0
                else:
                    # If some players have unknown cards, use a default value
                    current.state_value = 0.0

                current.processed = True
                continue

            # If we haven't processed this state yet, calculate its strategy
            if not current.info_state:
                player = current.game_state.current_player
                current.player = player

                # Create information state
                current.info_state = InfoState(
                    player_id=player,
                    hole_cards=current.game_state.player_cards[player],
                    community_cards=current.game_state.community_cards,
                    action_history=current.game_state.action_history,
                    street=current.game_state.street
                )

                # Get the current strategy for this info state
                current.strategy = self.get_strategy(current.info_state)

                # Update the strategy sum for calculating average strategy
                self.update_strategy_sum(current.info_state, current.strategy, current.reach_probs[player])

                # Get legal actions for this state
                legal_actions = current.game_state.get_legal_actions()
                current.pending_actions = [action for action, _, _ in legal_actions]

            # If no pending actions, this state is processed
            if not current.pending_actions:
                current.processed = True
                current.state_value = 0.0
                continue

            # Process the next pending action
            action = current.pending_actions[-1]  # Don't pop yet, wait until child returns

            # Create the next state
            next_state = GameState(
                num_players=current.game_state.num_players,
                stack_size=self.rules.stack_size,
                small_blind=self.rules.small_blind,
                big_blind=self.rules.big_blind
            )

            # Copy relevant state
            next_state.pot = current.game_state.pot
            next_state.street = current.game_state.street
            next_state.current_player = current.game_state.current_player
            next_state.player_cards = [cards.copy() if cards else [] for cards in current.game_state.player_cards]
            next_state.community_cards = current.game_state.community_cards.copy()
            next_state.action_history = current.game_state.action_history.copy() if current.game_state.action_history else []

            # Find the min amount for the action (for bets/raises)
            min_amount = 0.0
            for legal_action, legal_min, _ in current.game_state.get_legal_actions():
                if legal_action == action:
                    min_amount = legal_min
                    break

            # Apply the action
            try:
                street_complete = next_state.apply_action(action, min_amount)

                # If the street is complete, deal new cards
                if street_complete and not next_state.is_terminal():
                    if next_state.street == Street.PREFLOP:
                        # Deal flop (use random cards for prototype)
                        random_cards = self.rules.get_random_cards(3, [])
                        next_state.deal_community_cards(random_cards, Street.FLOP)
                    elif next_state.street == Street.FLOP:
                        # Deal turn (use random card)
                        random_card = self.rules.get_random_cards(1, next_state.community_cards)
                        next_state.deal_community_cards(random_card, Street.TURN)
                    elif next_state.street == Street.TURN:
                        # Deal river (use random card)
                        random_card = self.rules.get_random_cards(1, next_state.community_cards)
                        next_state.deal_community_cards(random_card, Street.RIVER)
            except Exception as e:
                self.logger.warning(f"Error applying action {action}: {e}")
                current.pending_actions.pop()  # Skip this action
                continue

            # Update reach probabilities for the next state
            next_reach_probs = current.reach_probs.copy()
            player = current.player
            if action in current.strategy:
                next_reach_probs[player] *= current.strategy[action]
            else:
                # Fall back to uniform strategy if action not in strategy (shouldn't happen)
                next_reach_probs[player] *= 1.0 / len(current.pending_actions)

            # Push the next state onto the stack
            stack.append(TraversalState(next_state, next_reach_probs, player))

        # We should never reach here - the stack should always have a final value
        return 0.0

    def iterate(self) -> Dict[str, Any]:
        """
        Perform one iteration of the DRM algorithm.

        Returns:
            Dictionary with metrics about the iteration
        """
        # Create a new random game state for this iteration
        hole_cards, _ = self.rules.create_random_game(num_players=2)

        # Initialize the game state
        state = GameState(
            num_players=2,
            stack_size=self.rules.stack_size,
            small_blind=self.rules.small_blind,
            big_blind=self.rules.big_blind
        )

        # Run the non-recursive CFR
        player_0_value = self.cfr_traverse(hole_cards, state)

        # Update iteration counter
        self.iteration_count += 1

        # Compute convergence metric (a simple one for the prototype)
        num_info_states = len(self.cumulative_regrets)
        avg_regret = 0.0
        if num_info_states > 0:
            total_regret = sum(
                sum(max(0.0, regret) for regret in regrets.values())
                for regrets in self.cumulative_regrets.values()
            )
            avg_regret = total_regret / num_info_states

        return {
            "iteration": self.iteration_count,
            "num_info_states": num_info_states,
            "player_0_value": player_0_value,
            "convergence": avg_regret
        }

    def compute_exploitability(self) -> float:
        """
        Compute the exploitability of the current strategy.

        Exploitability measures how far a strategy is from the Nash equilibrium.
        Lower values indicate a strategy closer to GTO.

        Returns:
            Exploitability value (lower is better)
        """
        # This is a simplified placeholder implementation
        # Computing true exploitability is complex and computationally expensive

        # For the prototype, we'll use a heuristic based on average positive regret
        total_positive_regret = 0.0
        num_states = max(1, len(self.cumulative_regrets))

        for regrets in self.cumulative_regrets.values():
            total_positive_regret += sum(max(0.0, regret) for regret in regrets.values())

        return total_positive_regret / num_states

    def save_model(self, path: Path) -> None:
        """
        Save the DRM model to disk.

        Args:
            path: Path to save the model
        """
        # Create the directory if it doesn't exist
        path.mkdir(parents=True, exist_ok=True)

        # Save the model data
        model_data = {
            "cumulative_regrets": self.cumulative_regrets,
            "strategy_sums": self.strategy_sums,
            "iteration_count": self.iteration_count,
            "discount_factor": self.discount_factor
        }

        # Save to disk
        with open(path / "drm_model.pkl", "wb") as f:
            pickle.dump(model_data, f)

        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: Path) -> None:
        """
        Load a DRM model from disk.

        Args:
            path: Path to the saved model
        """
        model_path = path / "drm_model.pkl"

        if not model_path.exists():
            self.logger.error(f"Model file not found at {model_path}")
            return

        try:
            # Load the model data
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)

            # Update the model attributes
            self.cumulative_regrets = model_data["cumulative_regrets"]
            self.strategy_sums = model_data["strategy_sums"]
            self.iteration_count = model_data["iteration_count"]
            self.discount_factor = model_data["discount_factor"]

            self.logger.info(f"Model loaded from {path}")
            self.logger.info(f"Loaded model with {len(self.cumulative_regrets)} info states")
            self.logger.info(f"Iteration count: {self.iteration_count}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")