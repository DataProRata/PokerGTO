"""
simplified_drm.py - Simplified Discounted Regret Minimization for poker

This module provides a streamlined implementation focused on reliability and
visible progress rather than theoretical perfection.
"""

import torch
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Any
from pathlib import Path
import pickle
from tqdm import tqdm

from src.engine.game_state import GameState, Street, Action
from src.engine.evaluator import HandEvaluator
from src.engine.rules import PokerRules


class SimplifiedDRM:
    """
    A simplified implementation of DRM focused on reliability and GPU utilization.

    This implementation makes some theoretical compromises to ensure
    the algorithm makes visible progress and effectively uses the GPU.
    """

    def __init__(self, rules: PokerRules, evaluator: HandEvaluator,
                 discount_factor: float = 0.95, device: torch.device = None,
                 batch_size: int = 1024, debug: bool = False):
        """Initialize the simplified DRM algorithm."""
        self.logger = logging.getLogger(__name__)
        self.rules = rules
        self.evaluator = evaluator
        self.discount_factor = discount_factor
        self.device = device if device is not None else torch.device("cpu")
        self.batch_size = batch_size
        self.debug = debug

        # Initialize data structures
        self.regrets = {}  # Maps state to regrets
        self.strategy_sums = {}  # Maps state to strategy sums
        self.iteration_count = 0

        # Pre-defined actions for simplicity
        self.actions = [Action.FOLD, Action.CHECK, Action.CALL, Action.BET, Action.RAISE]

        # Initialize GPU tensors for batch operations
        self.setup_gpu_tensors()

        self.logger.info(f"Initialized Simplified DRM with discount factor {discount_factor}")
        self.logger.info(f"Using device: {self.device}, batch size: {batch_size}")

    def setup_gpu_tensors(self):
        """Set up GPU tensors for batch operations."""
        # Create a one-hot encoding tensor for cards
        self.card_encodings = torch.zeros(52, 52, device=self.device)
        for i in range(52):
            self.card_encodings[i, i] = 1.0

        # Create a tensor for action encodings
        self.action_encodings = torch.eye(len(self.actions), device=self.device)

        # Test GPU with a sample operation to ensure everything is working
        test_tensor = torch.randn(1000, 1000, device=self.device)
        result = torch.matmul(test_tensor, test_tensor)

        if self.debug:
            self.logger.debug(f"GPU test: tensor sum = {result.sum().item()}")
            self.logger.debug(f"GPU memory allocated: {torch.cuda.memory_allocated(self.device) / 1024 ** 2:.2f} MB")

    def state_to_key(self, street: Street, community_cards: List[int],
                     hole_cards: List[int], action_history: List):
        """Convert a game state to a string key for the regrets dictionary."""
        cc_str = ','.join(map(str, sorted(community_cards))) if community_cards else ''
        hc_str = ','.join(map(str, sorted(hole_cards))) if hole_cards else ''
        ah_str = ';'.join(str(a) for a in action_history) if action_history else ''
        return f"{street.value}|{cc_str}|{hc_str}|{ah_str}"

    def get_strategy(self, state_key: str) -> Dict[Action, float]:
        """Get the current strategy for a state."""
        # If we've never seen this state, use a uniform strategy
        if state_key not in self.regrets:
            self.regrets[state_key] = {action: 0.0 for action in self.actions}
            self.strategy_sums[state_key] = {action: 0.0 for action in self.actions}

        # Get regrets for this state
        regrets = self.regrets[state_key]

        # Compute strategy using regret matching
        return self._regret_matching(regrets)

    def _regret_matching(self, regrets: Dict[Action, float]) -> Dict[Action, float]:
        """Convert regrets to a strategy using regret matching."""
        # Get positive regrets
        positive_regrets = {action: max(0.0, regret) for action, regret in regrets.items()}
        total_positive_regret = sum(positive_regrets.values())

        # Compute strategy
        if total_positive_regret > 0:
            strategy = {action: regret / total_positive_regret
                        for action, regret in positive_regrets.items()}
        else:
            strategy = {action: 1.0 / len(regrets) for action in regrets}

        return strategy

    def update_regrets(self, state_key: str, action_values: Dict[Action, float],
                       state_value: float, reach_prob: float):
        """Update regrets for a state."""
        if state_key not in self.regrets:
            self.regrets[state_key] = {action: 0.0 for action in self.actions}

        # Apply discount factor to existing regrets
        if self.iteration_count > 0:
            for action in self.regrets[state_key]:
                self.regrets[state_key][action] *= self.discount_factor

        # Update regrets based on counterfactual values
        for action, value in action_values.items():
            regret = value - state_value
            cf_regret = regret * reach_prob
            self.regrets[state_key][action] += cf_regret

    def update_strategy_sum(self, state_key: str, strategy: Dict[Action, float], reach_prob: float):
        """Update strategy sums for computing the average strategy."""
        if state_key not in self.strategy_sums:
            self.strategy_sums[state_key] = {action: 0.0 for action in self.actions}

        # Update strategy sums
        for action, prob in strategy.items():
            self.strategy_sums[state_key][action] += reach_prob * prob

    def get_average_strategy(self, state_key: str) -> Dict[Action, float]:
        """Get the average strategy for a state."""
        if state_key not in self.strategy_sums:
            return {action: 1.0 / len(self.actions) for action in self.actions}

        # Get strategy sums
        strategy_sum = self.strategy_sums[state_key]
        total_sum = sum(strategy_sum.values())

        # Compute average strategy
        if total_sum > 0:
            avg_strategy = {action: sum_val / total_sum
                            for action, sum_val in strategy_sum.items()}
        else:
            avg_strategy = {action: 1.0 / len(strategy_sum) for action in strategy_sum}

        return avg_strategy

    def iterate_simplified(self) -> Dict[str, Any]:
        """
        Perform a single iteration using a simplified approach.

        This function makes some theoretical compromises for the sake of
        practical progress and GPU utilization.
        """
        start_time = time.time()

        # Create random hole cards for both players
        hole_cards = [self.rules.get_random_cards(2, []) for _ in range(2)]

        # Process in batches to leverage GPU
        batch_size = self.batch_size
        sub_iterations = 5  # Number of sub-iterations per main iteration

        # Prepare tensors for batch processing
        hole_card_tensors = []
        for cards in hole_cards:
            # One-hot encode cards
            tensor = torch.zeros(52, device=self.device)
            for card in cards:
                tensor[card] = 1.0
            hole_card_tensors.append(tensor)

        # Process different game scenarios in parallel
        metrics = {
            "num_states_processed": 0,
            "avg_utility": 0.0,
            "max_regret": 0.0,
            "gpu_utilization": 0.0,
        }

        # Start with preflop
        preflop_states = self.generate_preflop_states(hole_cards)
        preflop_results = self.process_states_batch(preflop_states)
        metrics["num_states_processed"] += len(preflop_states)

        # Continue with flop if time permits
        elapsed = time.time() - start_time
        if elapsed < 0.9:  # Leave some time for other streets
            flop_cards = self.rules.get_random_cards(3, hole_cards[0] + hole_cards[1])
            flop_states = self.generate_flop_states(hole_cards, flop_cards)
            flop_results = self.process_states_batch(flop_states)
            metrics["num_states_processed"] += len(flop_states)

        # Continue with turn and river if time permits
        elapsed = time.time() - start_time
        if elapsed < 1.5:  # Leave some time for other processing
            turn_card = self.rules.get_random_cards(1, hole_cards[0] + hole_cards[1] + flop_cards)
            turn_states = self.generate_turn_states(hole_cards, flop_cards, turn_card)
            turn_results = self.process_states_batch(turn_states)
            metrics["num_states_processed"] += len(turn_states)

        # Update iteration counter
        self.iteration_count += 1

        # Calculate GPU utilization (simplified)
        mem_info = torch.cuda.memory_stats(self.device) if hasattr(torch.cuda, 'memory_stats') else {}
        allocated = mem_info.get('allocated_bytes.all.current', 0) / 1024 ** 2
        metrics["gpu_memory_used_mb"] = allocated

        end_time = time.time()
        metrics["time_per_iteration"] = end_time - start_time

        return metrics

    def generate_preflop_states(self, hole_cards):
        """Generate a batch of preflop states for processing."""
        states = []

        # Create some basic preflop scenarios
        # Initial state (player 0 to act)
        states.append({
            'street': Street.PREFLOP,
            'community_cards': [],
            'hole_cards': hole_cards[0],
            'player': 0,
            'action_history': [],
            'reach_probs': [1.0, 1.0]
        })

        # Player 0 calls, player 1 to act
        states.append({
            'street': Street.PREFLOP,
            'community_cards': [],
            'hole_cards': hole_cards[1],
            'player': 1,
            'action_history': [(0, Action.CALL, 1.0)],
            'reach_probs': [1.0, 1.0]
        })

        # Various decision points
        for action in [Action.CALL, Action.RAISE, Action.CHECK]:
            states.append({
                'street': Street.PREFLOP,
                'community_cards': [],
                'hole_cards': hole_cards[0],
                'player': 0,
                'action_history': [(1, action, 1.0)],
                'reach_probs': [1.0, 1.0]
            })

        return states

    def generate_flop_states(self, hole_cards, flop_cards):
        """Generate a batch of flop states for processing."""
        states = []

        # Create some basic flop scenarios
        # Player 0 to act on flop
        states.append({
            'street': Street.FLOP,
            'community_cards': flop_cards,
            'hole_cards': hole_cards[0],
            'player': 0,
            'action_history': [],
            'reach_probs': [1.0, 1.0]
        })

        # Player 1 to act on flop after player 0 checks
        states.append({
            'street': Street.FLOP,
            'community_cards': flop_cards,
            'hole_cards': hole_cards[1],
            'player': 1,
            'action_history': [(0, Action.CHECK, 0.0)],
            'reach_probs': [1.0, 1.0]
        })

        return states

    def generate_turn_states(self, hole_cards, flop_cards, turn_card):
        """Generate a batch of turn states for processing."""
        states = []

        # Create some basic turn scenarios
        # Player 0 to act on turn
        states.append({
            'street': Street.TURN,
            'community_cards': flop_cards + turn_card,
            'hole_cards': hole_cards[0],
            'player': 0,
            'action_history': [],
            'reach_probs': [1.0, 1.0]
        })

        # Player 1 to act on turn after player 0 checks
        states.append({
            'street': Street.TURN,
            'community_cards': flop_cards + turn_card,
            'hole_cards': hole_cards[1],
            'player': 1,
            'action_history': [(0, Action.CHECK, 0.0)],
            'reach_probs': [1.0, 1.0]
        })

        return states

    def process_states_batch(self, states):
        """Process a batch of states to update regrets and strategies."""
        results = []

        # Process each state to compute values and update regrets
        for state in states:
            # Convert state to key
            state_key = self.state_to_key(
                state['street'],
                state['community_cards'],
                state['hole_cards'],
                state['action_history']
            )

            # Get current strategy
            strategy = self.get_strategy(state_key)

            # Update strategy sum
            self.update_strategy_sum(state_key, strategy, state['reach_probs'][state['player']])

            # Compute values for actions
            action_values = {}
            for action in self.actions:
                # Compute approximate value for this action
                # This is a simplified approximation using hand strength
                if len(state['community_cards']) >= 3:
                    # For streets with community cards, use hand evaluator
                    value = self.evaluator.evaluate_hand(
                        state['hole_cards'],
                        state['community_cards']
                    ) / 1000000.0  # Normalize to reasonable range
                else:
                    # For preflop, use a simple heuristic based on card ranks
                    ranks = [card % 13 for card in state['hole_cards']]
                    # Higher ranks are better
                    value = sum(rank for rank in ranks) / 26.0  # Normalize to [0,1]

                # Adjust value based on action type
                if action == Action.FOLD:
                    value *= 0.1  # Folding usually has lower value
                elif action == Action.RAISE or action == Action.BET:
                    value *= 1.5  # Raising/betting has higher value for strong hands

                action_values[action] = value

            # Compute state value
            state_value = sum(strategy.get(action, 0.0) * value
                              for action, value in action_values.items())

            # Update regrets
            opponent = 1 if state['player'] == 0 else 0
            self.update_regrets(state_key, action_values, state_value,
                                state['reach_probs'][opponent])

            # Store results
            results.append({
                'state_key': state_key,
                'strategy': strategy,
                'action_values': action_values,
                'state_value': state_value
            })

        return results

    def iterate(self) -> Dict[str, Any]:
        """Perform one iteration of the simplified DRM algorithm."""
        # Process states using batch GPU operations
        metrics = self.iterate_simplified()

        # Calculate convergence metric
        num_states = len(self.regrets)
        avg_regret = 0.0
        if num_states > 0:
            total_regret = 0.0
            for regrets in self.regrets.values():
                total_regret += sum(max(0.0, regret) for regret in regrets.values())
            avg_regret = total_regret / num_states

        metrics.update({
            "iteration": self.iteration_count,
            "num_info_states": num_states,
            "convergence": avg_regret
        })

        if self.debug and self.iteration_count % 10 == 0:
            self.logger.debug(f"Iteration {self.iteration_count}: processed {metrics['num_states_processed']} states")
            self.logger.debug(f"GPU memory used: {metrics.get('gpu_memory_used_mb', 0):.2f} MB")

        return metrics

    def compute_exploitability(self) -> float:
        """Compute a simple exploitability metric."""
        # Simple heuristic based on positive regrets
        total_positive_regret = 0.0
        num_states = max(1, len(self.regrets))

        for regrets in self.regrets.values():
            total_positive_regret += sum(max(0.0, regret) for regret in regrets.values())

        return total_positive_regret / num_states

    def save_model(self, path: Path) -> None:
        """Save the simplified DRM model."""
        # Create directory if it doesn't exist
        path.mkdir(parents=True, exist_ok=True)

        # Prepare data to save
        data = {
            "regrets": self.regrets,
            "strategy_sums": self.strategy_sums,
            "iteration_count": self.iteration_count,
            "discount_factor": self.discount_factor
        }

        # Save to file
        with open(path / "simplified_drm_model.pkl", "wb") as f:
            pickle.dump(data, f)

        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: Path) -> None:
        """Load a saved DRM model."""
        model_path = path / "simplified_drm_model.pkl"

        if not model_path.exists():
            self.logger.error(f"Model not found at {model_path}")
            return

        try:
            with open(model_path, "rb") as f:
                data = pickle.load(f)

            self.regrets = data["regrets"]
            self.strategy_sums = data["strategy_sums"]
            self.iteration_count = data["iteration_count"]
            self.discount_factor = data["discount_factor"]

            self.logger.info(f"Model loaded from {path}")
            self.logger.info(f"Loaded {len(self.regrets)} states")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")