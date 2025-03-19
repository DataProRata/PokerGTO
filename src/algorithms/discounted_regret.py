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
from typing import Dict, List, Tuple, Optional, Set, Any
from pathlib import Path
import pickle
import time

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
                 community_cards: List[int], action_history: List[Tuple]):
        """
        Initialize an information state.

        Args:
            player_id: ID of the player viewing this state
            hole_cards: The player's hole cards
            community_cards: Visible community cards
            action_history: History of actions that led to this state
        """
        self.player_id = player_id
        self.hole_cards = sorted(hole_cards)
        self.community_cards = sorted(community_cards)
        self.action_history = action_history

    def __hash__(self) -> int:
        """Generate a hash for the information state."""
        # Create a hashable representation of the state
        hole_cards_tuple = tuple(self.hole_cards)
        community_cards_tuple = tuple(self.community_cards)
        action_history_tuple = tuple(
            (p, a.value if isinstance(a, Action) else a, float(amt))
            for p, a, amt in self.action_history
        )

        return hash((self.player_id, hole_cards_tuple, community_cards_tuple, action_history_tuple))

    def __eq__(self, other) -> bool:
        """Check if two information states are equal."""
        if not isinstance(other, InfoState):
            return False

        return (self.player_id == other.player_id and
                self.hole_cards == other.hole_cards and
                self.community_cards == other.community_cards and
                self.action_history == other.action_history)

    def __str__(self) -> str:
        """Return a string representation of the information state."""
        hole_str = ','.join(str(card) for card in self.hole_cards)
        comm_str = ','.join(str(card) for card in self.community_cards)
        action_str = '; '.join(
            f"Player {p}: {a.name if isinstance(a, Action) else a} {amt}"
            for p, a, amt in self.action_history
        )

        return f"Player {self.player_id} | Hole: [{hole_str}] | Community: [{comm_str}] | Actions: [{action_str}]"


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
        self.strategy_sums = {}  # Maps info state to cumulative strategy profile
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

        # Set hole cards
        player_cards = [[] for _ in range(game_state.num_players)]
        player_cards[info_state.player_id] = info_state.hole_cards
        game_state.player_cards = player_cards

        # Set community cards and determine street
        game_state.community_cards = info_state.community_cards
        if len(info_state.community_cards) == 0:
            game_state.street = Street.PREFLOP
        elif len(info_state.community_cards) == 3:
            game_state.street = Street.FLOP
        elif len(info_state.community_cards) == 4:
            game_state.street = Street.TURN
        elif len(info_state.community_cards) == 5:
            game_state.street = Street.RIVER

        # Replay action history to build the game state
        # For the prototype, this is simplified
        game_state.current_player = info_state.player_id

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

        # Update strategy sums weighted by reach probability
        for action, prob in strategy.items():
            if info_state_hash in self.strategy_sums and action in self.strategy_sums[info_state_hash]:
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

    def compute_cfr(self, state: GameState, reach_probs: List[float]) -> float:
        """
        Perform counterfactual regret minimization traversal.

        This is the core recursive algorithm that traverses the game tree
        and updates regrets and strategies.

        Args:
            state: Current game state
            reach_probs: Reach probabilities for each player

        Returns:
            Expected value of the state for the current player
        """
        player = state.current_player

        # If the state is terminal, return payoffs
        if state.is_terminal():
            # For terminal states, evaluate the hand strength
            hole_cards = [cards for cards in state.player_cards if cards]
            if len(hole_cards) >= 2:  # Need at least 2 players with hole cards
                hand_strengths = [
                    self.evaluator.evaluate_hand(cards, state.community_cards)
                    for cards in hole_cards
                ]
                payoffs = state.get_payoffs(hand_strengths)
                return payoffs[player]
            return 0.0

        # Create information state for the current player
        info_state = InfoState(
            player_id=player,
            hole_cards=state.player_cards[player],
            community_cards=state.community_cards,
            action_history=state.action_history
        )

        # Get the current strategy for this info state
        strategy = self.get_strategy(info_state)

        # Update the strategy sum (for computing the average strategy)
        self.update_strategy_sum(info_state, strategy, reach_probs[player])

        # Initialize action values
        action_values = {}
        legal_actions = state.get_legal_actions()

        # Compute counterfactual values for each action
        for action, min_amount, max_amount in legal_actions:
            # Clone the state for simulation
            next_state = GameState(
                num_players=state.num_players,
                stack_size=self.rules.stack_size,
                small_blind=self.rules.small_blind,
                big_blind=self.rules.big_blind
            )
            # Copy relevant state
            next_state.pot = state.pot
            next_state.street = state.street
            next_state.current_player = state.current_player
            next_state.player_cards = state.player_cards.copy()
            next_state.community_cards = state.community_cards.copy()

            # Apply the action
            # For simplicity, using min_amount for bets/raises in the prototype
            next_state.apply_action(action, min_amount)

            # Update reach probabilities
            next_reach_probs = reach_probs.copy()
            next_reach_probs[player] *= strategy[action]

            # Recursively compute the value
            action_values[action] = self.compute_cfr(next_state, next_reach_probs)

        # Compute the expected value of the state under the current strategy
        state_value = sum(strategy[action] * value for action, value in action_values.items())

        # Compute and update counterfactual regrets
        info_state_hash = hash(info_state)
        for action in action_values:
            # Regret = counterfactual value of action - expected value of state
            regret = action_values[action] - state_value

            # Counterfactual regret = reach probability of opponent * regret
            cf_regret = regret * reach_probs[1 - player]

            # Apply the discount factor to the current regret
            if self.iteration_count > 0:
                self.cumulative_regrets[info_state_hash][action] *= self.discount_factor

            # Update the cumulative regret
            self.cumulative_regrets[info_state_hash][action] += cf_regret

        return state_value

    def iterate(self) -> Dict[str, Any]:
        """
        Perform one iteration of the DRM algorithm.

        Returns:
            Dictionary with metrics about the iteration
        """
        # Create a new random game state for this iteration
        hole_cards, community_cards = self.rules.create_random_game(num_players=2)

        # Initialize the game state
        state = GameState(
            num_players=2,
            stack_size=self.rules.stack_size,
            small_blind=self.rules.small_blind,
            big_blind=self.rules.big_blind
        )

        # Set the cards
        state.deal_hole_cards(hole_cards)

        # Initialize reach probabilities
        reach_probs = [1.0, 1.0]

        # Run CFR traversal for each player
        player_0_value = self.compute_cfr(state, reach_probs)

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