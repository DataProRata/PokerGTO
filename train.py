#!/usr/bin/env python
"""
Collaborative Training Script for Poker AI Models
Implements advanced multi-model training with knowledge transfer
"""

import os
import sys
import time
import random
import numpy as np
import torch
from typing import List, Dict, Any
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.algorithms.collaborative_drm import CollaborativeDRM
from src.api.slumbot import SlumbotClient
from src.engine.game_state import GameState
from src.engine.evaluator import HandEvaluator
from src.engine.rules import PokerRules
from src.utils.knowledge_transfer import KnowledgeTransferManager
import config
import logging


class CollaborativeTrainer:
    def __init__(
            self,
            num_models: int = 6,
            games_per_model: int = 1000,
            knowledge_transfer_frequency: int = 100
    ):
        """
        Initialize collaborative training system.

        Args:
            num_models: Number of models to train simultaneously
            games_per_model: Number of games each model will play
            knowledge_transfer_frequency: How often to perform knowledge transfer
        """
        # Logging setup
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Game and learning components
        self.rules = PokerRules(
            small_blind=config.GAME_PARAMS['small_blind'],
            big_blind=config.GAME_PARAMS['big_blind'],
            stack_size=config.GAME_PARAMS['stack_size']
        )
        self.evaluator = HandEvaluator()

        # Knowledge transfer manager
        self.knowledge_manager = KnowledgeTransferManager()

        # Training parameters
        self.num_models = num_models
        self.games_per_model = games_per_model
        self.knowledge_transfer_frequency = knowledge_transfer_frequency

        # Slumbot client for training
        self.slumbot_client = SlumbotClient(
            base_url=config.SLUMBOT_API['base_url']
        )

        # Initialize models with diverse initial conditions
        self.models = self._create_diverse_models()

        # Output directory for trained models
        self.output_dir = Path(config.MODELS_DIR) / 'collaborative_models'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _create_diverse_models(self) -> List[CollaborativeDRM]:
        """
        Create a diverse set of initial models.

        Returns:
            List of initialized models
        """
        models = []
        for _ in range(self.num_models):
            model = CollaborativeDRM(
                rules=self.rules,
                evaluator=self.evaluator,
                discount_factor=random.uniform(0.9, 0.99),
                learning_rate=random.uniform(0.001, 0.1),
                exploration_factor=random.uniform(0.1, 0.5)
            )
            models.append(model)
        return models

    def train(self):
        """
        Collaborative training process.
        """
        self.logger.info(f"Starting collaborative training with {self.num_models} models")

        # Training loop
        for game_round in range(1, self.games_per_model + 1):
            # Parallel game simulation for all models
            round_results = self._simulate_round()

            # Periodic knowledge transfer
            if game_round % self.knowledge_transfer_frequency == 0:
                self._perform_knowledge_transfer(round_results)

            # Periodic logging and model saving
            if game_round % 100 == 0:
                self._log_round_summary(game_round, round_results)
                self._save_models(game_round)

        # Final model saving and statistics
        self._save_final_models()
        self._generate_training_report()

    def _simulate_round(self) -> List[Dict[str, Any]]:
        """
        Simulate a round of games for all models.

        Returns:
            List of game results for each model
        """
        round_results = []

        for model_index, model in enumerate(self.models):
            try:
                # Start a new hand against Slumbot
                hand_result = self._play_against_slumbot(model)

                # Update model's collaborative score
                model.collaborative_score += hand_result['win_value']

                # Store round result
                round_results.append({
                    'model_index': model_index,
                    'result': hand_result,
                    'model': model
                })

            except Exception as e:
                self.logger.error(f"Error in game for model {model_index}: {e}")

        return round_results

    def _play_against_slumbot(self, model: CollaborativeDRM) -> Dict[str, Any]:
        """
        Play a hand against Slumbot with the given model.

        Args:
            model: Poker AI model to play

        Returns:
            Dictionary of game result
        """
        # Start a new hand
        hand_info = self.slumbot_client.start_new_hand()

        # Strategy function using the model
        def model_strategy(state: Dict[str, Any]) -> Tuple[str, Optional[float]]:
            # Convert Slumbot state to model's representation
            from src.algorithms.discounted_regret import InfoState
            info_state = self._convert_state_to_info_state(state)

            # Get model's strategy
            strategy = model.get_strategy(info_state)

            # Convert strategy to Slumbot action
            legal_actions = state['legal_actions']
            chosen_action = self._choose_action_from_strategy(strategy, legal_actions)

            return chosen_action

        # Play the hand
        result = self.slumbot_client.play_hand(model_strategy)

        # Compute win value
        win_value = 1.0 if result.get('winnings', 0) > 0 else 0.0

        return {
            'win_value': win_value,
            'hand_details': result
        }

    def _convert_state_to_info_state(self, slumbot_state: Dict[str, Any]) -> 'InfoState':
        """
        Convert Slumbot state to model's InfoState.

        Args:
            slumbot_state: State from Slumbot

        Returns:
            Converted InfoState
        """
        # Implement conversion logic
        pass  # Placeholder for actual implementation

    def _choose_action_from_strategy(
            self,
            strategy: Dict[Action, float],
            legal_actions: List[Tuple[str, float]]
    ) -> Tuple[str, Optional[float]]:
        """
        Choose an action based on the model's strategy.

        Args:
            strategy: Model's action strategy
            legal_actions: Available actions

        Returns:
            Chosen action and optional bet amount
        """
        # Implement action selection logic
        pass  # Placeholder for actual implementation

    def _perform_knowledge_transfer(self, round_results: List[Dict[str, Any]]):
        """
        Perform knowledge transfer between models.

        Args:
            round_results: Results from the current round
        """
        # Sort models by performance
        sorted_models = sorted(
            round_results,
            key=lambda x: x['result']['win_value'],
            reverse=True
        )

        # Transfer knowledge from top performers
        for i in range(len(sorted_models)):
            for j in range(i + 1, len(sorted_models)):
                source_model = sorted_models[i]['model']
                target_model = sorted_models[j]['model']

                self.knowledge_manager.transfer_knowledge(
                    source_model,
                    target_model
                )

    def _log_round_summary(self, game_round: int, round_results: List[Dict[str, Any]]):
        """
        Log summary of the training round.

        Args:
            game_round: Current game round
            round_results: Results from the current round
        """
        win_rates = [result['result']['win_value'] for result in round_results]

        self.logger.info(f"Round {game_round} Summary:")
        self.logger.info(f"  Average Win Rate: {np.mean(win_rates):.2f}")
        self.logger.info(f"  Win Rate Std Dev: {np.std(win_rates):.2f}")
        self.logger.info(f"  Best Win Rate: {max(win_rates):.2f}")
        