"""
Collaborative Discounted Regret Minimization (DRM) Algorithm

Extends traditional DRM with collaborative learning mechanisms.
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from src.engine.game_state import GameState, Street, Action
from src.engine.evaluator import HandEvaluator
from src.engine.rules import PokerRules


class CollaborativeDRM:
    def __init__(
            self,
            rules: PokerRules,
            evaluator: HandEvaluator,
            discount_factor: float = 0.95,
            learning_rate: float = 0.01,
            exploration_factor: float = 0.2,
            memory_size: int = 10000
    ):
        """
        Initialize Collaborative Discounted Regret Minimization model.

        Args:
            rules: Poker game rules
            evaluator: Hand strength evaluator
            discount_factor: Regret discount rate
            learning_rate: Rate of learning from experiences
            exploration_factor: Probability of exploratory actions
            memory_size: Size of experience replay buffer
        """
        self.rules = rules
        self.evaluator = evaluator

        # Core learning parameters
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.exploration_factor = exploration_factor

        # Knowledge storage
        self.cumulative_regrets = {}
        self.strategy_sums = {}
        self.experience_replay = []
        self.memory_size = memory_size

        # Meta-learning tracking
        self.iteration_count = 0
        self.collaborative_score = 0.0

        # Logging setup
        self.logger = logging.getLogger(__name__)

    def get_strategy(self, info_state: 'InfoState') -> Dict[Action, float]:
        """
        Compute strategy for a given information state with exploration.

        Args:
            info_state: Current game information state

        Returns:
            Probabilistic action strategy
        """
        info_state_hash = hash(info_state)

        # Initialize state if not seen before
        if info_state_hash not in self.cumulative_regrets:
            self._initialize_state(info_state)

        # Retrieve regrets for this state
        regrets = self.cumulative_regrets[info_state_hash]

        # Regret matching with exploration
        strategy = self._compute_strategy_with_exploration(regstrategy = self._compute_strategy_with_exploration(regrets)

        # Update strategy sums for averaging
        self._update_strategy_sums(info_state_hash, strategy)

        return strategy

    def _compute_strategy_with_exploration(self, regrets: Dict[Action, float]) -> Dict[Action, float]:
        """
        Compute strategy with exploration mechanism.

        Args:
            regrets: Cumulative regrets for actions

        Returns:
            Probabilistic action strategy
        """
        # Positive regret matching
        positive_regrets = {action: max(0.0, regret) for action, regret in regrets.items()}
        total_positive_regret = sum(positive_regrets.values())

        # Base strategy computation
        if total_positive_regret > 0:
            strategy = {
                action: regret / total_positive_regret
                for action, regret in positive_regrets.items()
            }
        else:
            # Uniform strategy if no positive regrets
            strategy = {action: 1.0 / len(regrets) for action in regrets}

        # Exploration mechanism
        if random.random() < self.exploration_factor:
            # Exploration: Introduce randomness
            exploration_strategy = {
                action: random.random()
                for action in strategy
            }
            total_exploration = sum(exploration_strategy.values())
            strategy = {
                action: (1 - self.exploration_factor) * strategy.get(action, 0) +
                        self.exploration_factor * (exploration_strategy[action] / total_exploration)
                for action in strategy
            }

        return strategy

    def _update_strategy_sums(self, info_state_hash: int, strategy: Dict[Action, float]):
        """
        Update cumulative strategy sums for averaging.

        Args:
            info_state_hash: Unique hash for the information state
            strategy: Current action strategy
        """
        if info_state_hash not in self.strategy_sums:
            self.strategy_sums[info_state_hash] = {
                action: 0.0 for action in strategy
            }

        for action, prob in strategy.items():
            self.strategy_sums[info_state_hash][action] += prob

    def _initialize_state(self, info_state: 'InfoState'):
        """
        Initialize a new information state.

        Args:
            info_state: Game information state to initialize
        """
        info_state_hash = hash(info_state)

        # Get legal actions for the state
        game_state = self._info_state_to_game_state(info_state)
        legal_actions = game_state.get_legal_actions()

        # Initialize regrets and strategy sums
        self.cumulative_regrets[info_state_hash] = {
            action[0]: 0.0 for action in legal_actions
        }
        self.strategy_sums[info_state_hash] = {
            action[0]: 0.0 for action in legal_actions
        }

    def update_regrets(self,
                       info_state: 'InfoState',
                       action_values: Dict[Action, float],
                       state_value: float):
        """
        Update regrets based on action values.

        Args:
            info_state: Current game information state
            action_values: Values for each possible action
            state_value: Overall state value
        """
        info_state_hash = hash(info_state)

        # Ensure state exists
        if info_state_hash not in self.cumulative_regrets:
            self._initialize_state(info_state)

        # Apply discount to existing regrets
        if self.iteration_count > 0:
            for action in self.cumulative_regrets[info_state_hash]:
                self.cumulative_regrets[info_state_hash][action] *= self.discount_factor

        # Update regrets
        for action, value in action_values.items():
            regret = value - state_value
            self.cumulative_regrets[info_state_hash][action] += regret

        # Manage experience replay memory
        self._update_experience_replay(info_state, action_values, state_value)

    def _update_experience_replay(self,
                                  info_state: 'InfoState',
                                  action_values: Dict[Action, float],
                                  state_value: float):
        """
        Store and manage experience replay buffer.

        Args:
            info_state: Current game information state
            action_values: Values for each possible action
            state_value: Overall state value
        """
        experience = {
            'info_state': info_state,
            'action_values': action_values,
            'state_value': state_value,
            'timestamp': self.iteration_count
        }

        # Add to experience replay
        self.experience_replay.append(experience)

        # Maintain memory size
        if len(self.experience_replay) > self.memory_size:
            # Remove oldest experiences
            self.experience_replay = self.experience_replay[-self.memory_size:]

    def learn_from_experience_replay(self):
        """
        Learn from stored experiences using replay mechanism.
        """
        if not self.experience_replay:
            return

        # Sample experiences (weighted towards recent experiences)
        sample_size = min(len(self.experience_replay), 100)

        # Weight recent experiences more heavily
        weights = [
            (self.iteration_count - exp['timestamp']) ** 2
            for exp in self.experience_replay[-sample_size:]
        ]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Weighted sampling
        sampled_experiences = random.choices(
            self.experience_replay[-sample_size:],
            weights=weights,
            k=sample_size // 2
        )

        # Learn from sampled experiences
        for experience in sampled_experiences:
            self.update_regrets(
                experience['info_state'],
                experience['action_values'],
                experience['state_value']
            )

    def get_collaborative_insights(self) -> Dict[str, Any]:
        """
        Generate insights about the model's learning.

        Returns:
            Dictionary of collaborative learning metrics
        """
        return {
            'iteration_count': self.iteration_count,
            'collaborative_score': self.collaborative_score,
            'unique_states': len(self.cumulative_regrets),
            'exploration_factor': self.exploration_factor,
            'learning_rate': self.learning_rate
        }

    def integrate_external_knowledge(self, external_knowledge: Dict[str, Any]):
        """
        Integrate knowledge from external sources.

        Args:
            external_knowledge: Knowledge dictionary to integrate
        """
        # Update collaborative score
        self.collaborative_score += external_knowledge.get('contribution_value', 0)

        # Selectively update regrets and strategies
        for state_hash, knowledge in external_knowledge.get('state_knowledge', {}).items():
            if state_hash not in self.cumulative_regrets:
                self.cumulative_regrets[state_hash] = {}

            # Weighted integration of external knowledge
            for action, ext_regret in knowledge.get('regrets', {}).items():
                current_regret = self.cumulative_regrets[state_hash].get(action, 0)
                self.cumulative_regrets[state_hash][action] = (
                    current_regret * (1 - self.learning_rate) +
                    ext_regret * self.learning_rate
                )

    def _info_state_to_game_state(self, info_state: 'InfoState') -> GameState:
        """
        Convert information state to game state.

        Args:
            info_state: Information state to convert

        Returns:
            Corresponding game state
        """
        # Implement conversion logic
        game_state = GameState(
            num_players=2,
            stack_size=self.rules.stack_size,
            small_blind=self.rules.small_blind,
            big_blind=self.rules.big_blind
        )

        # Populate game state based on info_state
        game_state.current_player = info_state.player_id
        game_state.player_cards[info_state.player_id] = info_state.hole_cards
        game_state.community_cards = info_state.community_cards
        game_state.action_history = info_state.action_history
        game_state.street = info_state.street

        return game_state

    def save_model(self, path: Path):
        """
        Save the collaborative model.

        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        path.mkdir(parents=True, exist_ok=True)

        # Prepare model data
        model_data = {
            'cumulative_regrets': self.cumulative_regrets,
            'strategy_sums': self.strategy_sums,
            'iteration_count': self.iteration_count,
            'collaborative_score': self.collaborative_score,
            'hyperparameters': {
                'discount_factor': self.discount_factor,
                'learning_rate': self.learning_rate,
                'exploration_factor': self.exploration_factor
            }
        }

        # Save using pickle
        import pickle
        with open(path / 'collaborative_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)

        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: Path):
        """
        Load a previously saved collaborative model.

        Args:
            path: Path to load the model from
        """
        import pickle

        model_path = path / 'collaborative_model.pkl'

        if not model_path.exists():
            self.logger.error(f"Model not found at {model_path}")
            return

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        # Restore model state
        self.cumulative_regrets = model_data['cumulative_regrets']
        self.strategy_sums = model_data['strategy_sums']
        self.iteration_count = model_data['iteration_count']
        self.collaborative_score = model_data['collaborative_score']

        # Restore hyperparameters
        hyperparams = model_data['hyperparameters']
        self.discount_factor = hyperparams['discount_factor']
        self.learning_rate = hyperparams['learning_rate']
        self.exploration_factor = hyperparams['exploration_factor']

        self.logger.info(f"Model loaded from {path}")
        self.logger.info(f"Loaded {len(self.cumulative_regrets)} unique states")