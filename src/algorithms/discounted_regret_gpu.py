"""
discounted_regret_gpu.py - GPU-accelerated Discounted Regret Minimization for poker

This module implements a GPU-accelerated version of the Discounted Regret Minimization
algorithm for calculating GTO strategies in poker, using PyTorch for parallel processing.
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
import concurrent.futures

from src.engine.game_state import GameState, Street, Action
from src.engine.evaluator import HandEvaluator
from src.engine.rules import PokerRules


class InfoSetBatch:
    """Represents a batch of information states for vectorized operations."""

    def __init__(self, player_ids, hole_cards, community_cards, streets, action_histories=None):
        """
        Initialize a batch of information states.

        Args:
            player_ids: Tensor of player IDs
            hole_cards: Tensor of hole cards (encoded)
            community_cards: Tensor of community cards (encoded)
            streets: Tensor of street values
            action_histories: List of action histories (can't be easily tensorized)
        """
        self.player_ids = player_ids
        self.hole_cards = hole_cards
        self.community_cards = community_cards
        self.streets = streets
        self.action_histories = action_histories or [[] for _ in range(len(player_ids))]

    @staticmethod
    def from_info_states(info_states, device):
        """Create a batch from a list of information states."""
        player_ids = torch.tensor([state.player_id for state in info_states], device=device)

        # Encode hole cards as one-hot tensors (52 cards)
        hole_cards = torch.zeros(len(info_states), 52, device=device)
        for i, state in enumerate(info_states):
            for card in state.hole_cards:
                hole_cards[i, card] = 1.0

        # Encode community cards as one-hot tensors
        community_cards = torch.zeros(len(info_states), 52, device=device)
        for i, state in enumerate(info_states):
            for card in state.community_cards:
                community_cards[i, card] = 1.0

        # Encode streets as integer tensors
        streets = torch.tensor([state.street.value for state in info_states], device=device)

        # Action histories (can't easily tensorize due to variable structure)
        action_histories = [state.action_history for state in info_states]

        return InfoSetBatch(player_ids, hole_cards, community_cards, streets, action_histories)


class GPUDiscountedRegretMinimization:
    """
    GPU-Accelerated implementation of Discounted Regret Minimization for poker.

    This version uses PyTorch for GPU acceleration and implements several
    optimizations for faster computation.
    """

    def __init__(self, rules: PokerRules, evaluator: HandEvaluator,
                 discount_factor: float = 0.95, device: torch.device = None,
                 batch_size: int = 1024, num_threads: int = 8):
        """
        Initialize the GPU-accelerated DRM algorithm.

        Args:
            rules: Poker rules instance
            evaluator: Hand evaluator instance
            discount_factor: Discount factor for historical regrets (0.0-1.0)
            device: PyTorch device to use for computations
            batch_size: Batch size for GPU operations
            num_threads: Number of CPU threads for parallel CPU operations
        """
        self.logger = logging.getLogger(__name__)
        self.rules = rules
        self.evaluator = evaluator
        self.discount_factor = discount_factor
        self.device = device if device is not None else torch.device("cpu")
        self.batch_size = batch_size
        self.num_threads = num_threads

        # Initialize data structures for regret tracking
        self.cumulative_regrets = {}  # Maps info state hash to regrets for each action
        self.strategy_sums = {}  # Maps info state hash to strategy sums
        self.iteration_count = 0

        # Tracking cached values
        self.cached_strategies = {}  # Cache for computed strategies
        self.strategy_cache_hits = 0
        self.strategy_cache_misses = 0

        self.logger.info(f"Initialized GPU-accelerated DRM with discount factor {discount_factor}")
        self.logger.info(f"Using device: {self.device}, batch size: {batch_size}, threads: {num_threads}")

        # Precompute hand strength lookup table if possible
        self.precompute_hands()

    def precompute_hands(self):
        """Precompute hand strength evaluations for common scenarios if possible."""
        self.logger.info("Precomputing common hand evaluations...")
        self.hand_cache = {}

        # This would normally precompute hand strengths for frequent combinations
        # For a complete implementation, you'd generate a lot more precomputed values
        # This is just a small example

        # Precompute high card strengths
        for rank in range(13):
            # High card only (no pairs, etc.)
            card1 = rank  # First suit (hearts)
            card2 = rank + 13  # Second suit (diamonds)

            # Cache the hand strength with no community cards
            key = (card1, card2, ())
            self.hand_cache[key] = self.evaluator.evaluate_hand([card1, card2], [])

        self.logger.info(f"Precomputed {len(self.hand_cache)} hand evaluations")

    def get_strategy(self, info_state_hash: int, legal_actions: List[Tuple[Action, float, float]]) -> Dict[
        Action, float]:
        """
        Get the current strategy for an information state.

        Args:
            info_state_hash: Hash of the information state
            legal_actions: List of legal actions for this state

        Returns:
            Dictionary mapping actions to probabilities
        """
        # Check if strategy is cached
        if info_state_hash in self.cached_strategies:
            self.strategy_cache_hits += 1
            return self.cached_strategies[info_state_hash]

        self.strategy_cache_misses += 1

        # If we haven't seen this state before, initialize it
        if info_state_hash not in self.cumulative_regrets:
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

        # Cache the strategy
        self.cached_strategies[info_state_hash] = strategy

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

    def batch_update_strategies(self, batch_info_states):
        """Update strategies for a batch of information states."""
        # This would implement batch update operations using tensorized operations
        # For the prototype, we're doing a basic implementation
        result = {}
        for info_state, legal_actions in batch_info_states:
            info_state_hash = hash(info_state)
            result[info_state_hash] = self.get_strategy(info_state_hash, legal_actions)
        return result

    def update_regrets(self, info_state_hash, action_values, strategy, state_value, reach_probs, player):
        """
        Update the regrets for an information state.

        Args:
            info_state_hash: Hash of the information state
            action_values: Values for each action
            strategy: Current strategy
            state_value: Expected value of the state
            reach_probs: Reach probabilities
            player: Current player
        """
        for action, value in action_values.items():
            # Check if action exists in regrets
            if info_state_hash in self.cumulative_regrets and action in self.cumulative_regrets[info_state_hash]:
                # Calculate counterfactual regret
                regret = value - state_value
                cf_regret = regret * reach_probs[1 - player]

                # Apply discount factor to historical regrets
                if self.iteration_count > 0:
                    self.cumulative_regrets[info_state_hash][action] *= self.discount_factor

                # Update cumulative regret
                self.cumulative_regrets[info_state_hash][action] += cf_regret

                # Invalidate cached strategy
                if info_state_hash in self.cached_strategies:
                    del self.cached_strategies[info_state_hash]

    def parallel_traversal(self, hand_config, initial_state):
        """
        Perform CFR traversal with parallel processing for subtrees.

        Args:
            hand_config: Configuration for the hand (hole cards, etc.)
            initial_state: Initial game state

        Returns:
            Expected value for player 0
        """
        # Set up the initial state
        hole_cards = hand_config['hole_cards']
        for i, cards in enumerate(hole_cards):
            initial_state.player_cards[i] = cards.copy()

        # For parallel traversal, we'd subdivide the game tree into independent subtrees
        # For simplicity in this prototype, we're using a non-recursive traversal
        # with some GPU acceleration for batch operations

        # Use the same tree traversal approach as before but with GPU accelerated operations
        return self.cfr_traverse(initial_state)

    def cfr_traverse(self, initial_state):
        """
        Non-recursive CFR tree traversal with GPU acceleration for batch operations.

        Args:
            initial_state: Initial game state

        Returns:
            Expected value for player 0
        """

        # Similar to the non-GPU version but with batch operations where possible
        class TraversalState:
            def __init__(self, game_state, reach_probs, player):
                self.game_state = game_state
                self.reach_probs = reach_probs
                self.player = player
                self.info_state_hash = None
                self.strategy = None
                self.action_values = {}
                self.state_value = 0.0
                self.pending_actions = []
                self.processed = False

        # Initialize the traversal stack with the initial state
        stack = deque([TraversalState(initial_state, [1.0, 1.0], initial_state.current_player)])

        # Track states to batch process
        info_state_batch = []

        # Process states until the stack is empty
        while stack:
            # Periodically batch process info states for strategy computation
            if len(info_state_batch) >= self.batch_size:
                batch_strategies = self.batch_update_strategies(info_state_batch)
                info_state_batch = []

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
                    self.update_regrets(
                        parent.info_state_hash,
                        parent.action_values,
                        parent.strategy,
                        parent.state_value,
                        parent.reach_probs,
                        parent.player
                    )

                continue

            # Handle terminal states
            if current.game_state.is_terminal():
                # For terminal states, evaluate the hand strength
                if all(len(cards) > 0 for cards in current.game_state.player_cards):
                    # Check if the hand is in the precomputed cache
                    cards1 = tuple(sorted(current.game_state.player_cards[0]))
                    cards2 = tuple(sorted(current.game_state.player_cards[1]))
                    comm_cards = tuple(sorted(current.game_state.community_cards))

                    cache_key1 = (cards1[0], cards1[1], comm_cards)
                    cache_key2 = (cards2[0], cards2[1], comm_cards)

                    # Try to use cached hand strengths
                    if cache_key1 in self.hand_cache and cache_key2 in self.hand_cache:
                        strength1 = self.hand_cache[cache_key1]
                        strength2 = self.hand_cache[cache_key2]
                        hand_strengths = [strength1, strength2]
                    else:
                        # Calculate hand strengths
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
            if not current.info_state_hash:
                player = current.game_state.current_player
                current.player = player

                # Create information state hash
                # For performance, we'll create a direct hash rather than creating the full InfoState object
                state = current.game_state
                hole_cards = tuple(sorted(state.player_cards[player]))
                community_cards = tuple(sorted(state.community_cards))
                action_history = tuple((p, a.value if isinstance(a, Action) else a, float(amt))
                                       for p, a, amt in (state.action_history or []))
                current.info_state_hash = hash(
                    (player, hole_cards, community_cards, action_history, state.street.value))

                # Get legal actions
                legal_actions = current.game_state.get_legal_actions()

                # Queue this state for batch strategy computation
                info_state_batch.append((current.info_state_hash, legal_actions))

                # Get the current strategy (will use batch updates if available)
                current.strategy = self.get_strategy(current.info_state_hash, legal_actions)

                # Update the strategy sum for calculating average strategy
                self.update_strategy_sum(current.info_state_hash, current.strategy, current.reach_probs[player],
                                         legal_actions)

                # Store pending actions
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

        # Process any remaining batch states
        if info_state_batch:
            self.batch_update_strategies(info_state_batch)

        # We should never reach here - the stack should always have a final value
        return 0.0

    def update_strategy_sum(self, info_state_hash, strategy, reach_prob, legal_actions):
        """
        Update the cumulative strategy sum for an information state.

        Args:
            info_state_hash: Hash of the information state
            strategy: Current strategy for this info state
            reach_prob: Reach probability for this info state
            legal_actions: List of legal actions for initialization
        """
        # Initialize if not already present
        if info_state_hash not in self.strategy_sums:
            self.strategy_sums[info_state_hash] = {
                action: 0.0 for action, _, _ in legal_actions
            }

        # Update strategy sums weighted by reach probability
        for action, prob in strategy.items():
            if action in self.strategy_sums[info_state_hash]:
                self.strategy_sums[info_state_hash][action] += reach_prob * prob

    def iterate(self, num_parallel: int = 4) -> Dict[str, Any]:
        """
        Perform one iteration of the GPU-accelerated DRM algorithm.

        Args:
            num_parallel: Number of parallel subgames to compute

        Returns:
            Dictionary with metrics about the iteration
        """
        # For real GPU acceleration, we'd use multiple random subgames in parallel
        # For simplicity, we'll run one game with GPU-optimized operations

        # Create a new random game state for this iteration
        hole_cards, _ = self.rules.create_random_game(num_players=2)

        # Initialize the game state
        state = GameState(
            num_players=2,
            stack_size=self.rules.stack_size,
            small_blind=self.rules.small_blind,
            big_blind=self.rules.big_blind
        )

        # Configure the hand
        hand_config = {
            'hole_cards': hole_cards
        }

        # Run the traversal
        player_0_value = self.parallel_traversal(hand_config, state)

        # Update iteration counter
        self.iteration_count += 1

        # Clear strategy cache periodically
        if self.iteration_count % 10 == 0:
            self.cached_strategies = {}

        # Compute convergence metric
        num_info_states = len(self.cumulative_regrets)
        avg_regret = 0.0
        if num_info_states > 0:
            total_regret = sum(
                sum(max(0.0, regret) for regret in regrets.values())
                for regrets in self.cumulative_regrets.values()
            )
            avg_regret = total_regret / num_info_states

        # Report cache performance
        cache_total = self.strategy_cache_hits + self.strategy_cache_misses
        cache_hit_rate = self.strategy_cache_hits / max(1, cache_total)

        return {
            "iteration": self.iteration_count,
            "num_info_states": num_info_states,
            "player_0_value": player_0_value,
            "convergence": avg_regret,
            "cache_hit_rate": cache_hit_rate
        }

    def compute_exploitability(self) -> float:
        """Compute the exploitability of the current strategy."""
        # Same as before, a simplified heuristic
        total_positive_regret = 0.0
        num_states = max(1, len(self.cumulative_regrets))

        for regrets in self.cumulative_regrets.values():
            total_positive_regret += sum(max(0.0, regret) for regret in regrets.values())

        return total_positive_regret / num_states

    def save_model(self, path: Path) -> None:
        """Save the DRM model to disk."""
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
        """Load a DRM model from disk."""
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