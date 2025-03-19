import os
import sys
import random
import numpy as np
import torch
import logging
from typing import List, Dict, Any
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.algorithms.collaborative_drm import CollaborativeDRM
from src.engine.game_state import GameState
from src.engine.evaluator import HandEvaluator
from src.engine.rules import PokerRules
from src.utils.knowledge_transfer import KnowledgeTransferManager
import config


class EvolutionaryPokerAI:
    def __init__(
            self,
            population_size: int = 12,  # Increased population
            mutation_rate: float = 0.15,
            crossover_rate: float = 0.75,
            knowledge_transfer_rate: float = 0.5
    ):
        """
        Advanced evolutionary poker AI with collaborative learning.

        Args:
            population_size: Number of AI models
            mutation_rate: Probability of random mutations
            crossover_rate: Probability of genetic crossover
            knowledge_transfer_rate: Rate of inter-model knowledge sharing
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.knowledge_transfer_rate = knowledge_transfer_rate

        # Setup logging
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

        # Create initial population with diverse initial conditions
        self.population = self._create_diverse_population()

        # Output directory for evolved models
        self.output_dir = Path(config.MODELS_DIR) / 'collaborative_models'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _create_diverse_population(self) -> List[CollaborativeDRM]:
        """
        Create a population with diverse initial characteristics.
        """
        population = []
        for i in range(self.population_size):
            # Vary key hyperparameters
            model = CollaborativeDRM(
                rules=self.rules,
                evaluator=self.evaluator,
                discount_factor=random.uniform(0.85, 0.99),
                learning_rate=random.uniform(0.001, 0.1),
                exploration_factor=random.uniform(0.1, 0.5)
            )
            population.append(model)
        return population

    def _advanced_tournament_selection(self, fitness_scores: List[float]) -> List[int]:
        """
        Advanced tournament selection with diversity preservation.
        """
        selected_indices = []
        while len(selected_indices) < self.population_size:
            # More sophisticated tournament selection
            tournament = random.sample(
                range(len(fitness_scores)),
                min(5, len(fitness_scores))
            )

            # Weight selection towards best performers but allow some randomness
            weighted_tournament = sorted(
                tournament,
                key=lambda i: fitness_scores[i],
                reverse=True
            )

            # Probabilistic selection favoring top performers
            winner = weighted_tournament[
                min(
                    random.choices(
                        range(len(weighted_tournament)),
                        weights=[max(len(weighted_tournament) - i, 1) for i in range(len(weighted_tournament))]
                    )[0],
                    len(weighted_tournament) - 1
                )
            ]

            selected_indices.append(winner)

        return selected_indices

    def _collaborative_knowledge_transfer(self, models: List[CollaborativeDRM]):
        """
        Advanced knowledge transfer between models.
        """
        for i in range(len(models)):
            if random.random() < self.knowledge_transfer_rate:
                # Select a knowledge donor model
                donor_index = random.choice(
                    [j for j in range(len(models)) if j != i]
                )

                # Transfer specific knowledge components
                self.knowledge_manager.transfer_knowledge(
                    source_model=models[donor_index],
                    target_model=models[i]
                )

    def evolve(self, generations: int = 20):
        """
        Advanced evolutionary training process.
        """
        for generation in range(generations):
            self.logger.info(f"\n--- Generation {generation + 1} ---")

            # Run comprehensive tournament
            fitness_scores = self._run_comprehensive_tournament()

            # Log fitness details
            self._log_generation_stats(generation, fitness_scores)

            # Select models for next generation
            selected_indices = self._advanced_tournament_selection(fitness_scores)

            # Create new population
            new_population = []
            while len(new_population) < self.population_size:
                # Select parents with weighted probability
                parent1_idx = random.choice(selected_indices)
                parent2_idx = random.choice(selected_indices)

                # Crossover and mutation
                child = self._advanced_reproduction(
                    self.population[parent1_idx],
                    self.population[parent2_idx]
                )

                new_population.append(child)

            # Collaborative knowledge transfer
            self._collaborative_knowledge_transfer(new_population)

            # Update population
            self.population = new_population

            # Save best model
            best_model_idx = fitness_scores.index(max(fitness_scores))
            best_model_path = self.output_dir / f'best_model_gen_{generation}.pkl'
            self.population[best_model_idx].save_model(best_model_path)

    def _run_comprehensive_tournament(self) -> List[float]:
        """
        Run a comprehensive multi-stage tournament.
        """
        fitness_scores = [0.0] * self.population_size

        # Multiple tournament stages with different game configurations
        tournament_stages = [
            {'num_games': 100, 'game_complexity': 'simple'},
            {'num_games': 50, 'game_complexity': 'medium'},
            {'num_games': 25, 'game_complexity': 'complex'}
        ]

        for stage in tournament_stages:
            stage_scores = self._stage_tournament(stage['num_games'], stage['game_complexity'])

            # Weighted accumulation of scores
            for i in range(self.population_size):
                fitness_scores[i] += stage_scores[i] * (
                    1 if stage['game_complexity'] == 'simple' else
                    1.5 if stage['game_complexity'] == 'medium' else
                    2
                )

        return fitness_scores

    def _stage_tournament(self, num_games: int, complexity: str) -> List[float]:
        """
        Single tournament stage with configurable complexity.
        """
        stage_scores = [0.0] * self.population_size

        for i in range(self.population_size):
            for j in range(i + 1, self.population_size):
                game_results = self._simulate_games(
                    self.population[i],
                    self.population[j],
                    num_games=num_games,
                    complexity=complexity
                )

                stage_scores[i] += game_results['player1_score']
                stage_scores[j] += game_results['player2_score']

        return stage_scores

    def _simulate_games(self,
                        model1: CollaborativeDRM,
                        model2: CollaborativeDRM,
                        num_games: int = 100,
                        complexity: str = 'simple') -> Dict[str, float]:
        """
        Simulate games with dynamic complexity.
        """
        # Complexity-based game configuration
        game_configs = {
            'simple': {'max_raises': 2, 'initial_stack': 100},
            'medium': {'max_raises': 4, 'initial_stack': 200},
            'complex': {'max_raises': 6, 'initial_stack': 500}
        }

        config = game_configs.get(complexity, game_configs['simple'])

        # Simulation metrics
        total_games = num_games
        player1_score = 0
        player2_score = 0

        for _ in range(total_games):
            game_state = GameState(
                num_players=2,
                stack_size=config['initial_stack'],
                small_blind=self.rules.small_blind,
                big_blind=self.rules.big_blind
            )

            # Adjust game rules based on complexity
            game_state.max_raises = config['max_raises']

            hole_cards = self.rules.deal_hole_cards(num_players=2)
            game_state.deal_hole_cards(hole_cards)

            # Advanced game simulation
            result = self._advanced_game_simulation(game_state, model1, model2)

            # Score accumulation with complexity weighting
            player1_score += result['player1_score']
            player2_score += result['player2_score']

        return {
            'player1_score': player1_score / total_games,
            'player2_score': player2_score / total_games
        }

    def _advanced_game_simulation(self,
                                  game_state: GameState,
                                  model1: CollaborativeDRM,
                                  model2: CollaborativeDRM) -> Dict[str, float]:
        """
        Advanced game simulation with collaborative strategy.
        """
        # Implement a more nuanced game simulation
        # This would involve more complex state tracking and strategy evaluation
        pass  # Placeholder for advanced implementation

    def _advanced_reproduction(self,
                               parent1: CollaborativeDRM,
                               parent2: CollaborativeDRM) -> CollaborativeDRM:
        """
        Advanced genetic reproduction with multiple mutation strategies.
        """
        # Crossover of model parameters
        child = CollaborativeDRM(
            rules=self.rules,
            evaluator=self.evaluator,
            discount_factor=(parent1.discount_factor + parent2.discount_factor) / 2,
            learning_rate=(parent1.learning_rate + parent2.learning_rate) / 2
        )

        # Multiple mutation strategies
        mutations = [
            self._parameter_mutation,
            self._structural_mutation,
            self._strategy_mutation
        ]

        for mutation in mutations:
            if random.random() < self.mutation_rate:
                child = mutation(child)

        return child

    def _parameter_mutation(self, model: CollaborativeDRM) -> CollaborativeDRM:
        """Mutate hyperparameters"""
        model.discount_factor *= random.uniform(0.9, 1.1)
        model.learning_rate *= random.uniform(0.9, 1.1)
        return model

    def _structural_mutation(self, model: CollaborativeDRM) -> CollaborativeDRM:
        """Structural changes to model architecture"""
        # Add placeholder for more complex architectural mutations
        return model

    def _strategy_mutation(self, model: CollaborativeDRM) -> CollaborativeDRM:
        """Mutate core strategy components"""
        # Add strategy-level mutations
        return model

    def _log_generation_stats(self, generation: int, fitness_scores: List[float]):
        """Log detailed statistics for each generation"""
        self.logger.info(f"Generation {generation + 1} Statistics:")
        self.logger.info(f"Best Fitness: {max(fitness_scores)}")
        self.logger.info(f"Average Fitness: {np.mean(fitness_scores)}")
        self.logger.info(f"Fitness Std Dev: {np.std(fitness_scores)}")


def main():
    # Create and run evolutionary training
    poker_evolution = EvolutionaryPokerAI(
        population_size=12,
        mutation_rate=0.15,
        crossover_rate=0.75,
        knowledge_transfer_rate=0.5
    )

    poker_evolution.evolve(generations=20)


if __name__ == "__main__":
    main()