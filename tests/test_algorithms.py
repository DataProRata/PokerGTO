import unittest
import sys
import os
import torch
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.engine.rules import PokerRules
from src.engine.evaluator import HandEvaluator
from src.algorithms.discounted_regret import DiscountedRegretMinimization, InfoState
from src.engine.game_state import Action, Street


class TestDiscountedRegretMinimization(unittest.TestCase):
    def setUp(self):
        """Setup test environment for DRM algorithm."""
        rules = PokerRules()
        evaluator = HandEvaluator()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.drm = DiscountedRegretMinimization(
            rules=rules,
            evaluator=evaluator,
            discount_factor=0.95,
            device=self.device
        )

    def test_info_state_creation(self):
        """Test creating an information state."""
        info_state = InfoState(
            player_id=0,
            hole_cards=[0, 1],
            community_cards=[2, 3, 4],
            action_history=[(0, Action.CALL, 1.0)],
            street=Street.FLOP
        )

        self.assertEqual(info_state.player_id, 0)
        self.assertEqual(info_state.hole_cards, [0, 1])
        self.assertEqual(info_state.community_cards, [2, 3, 4])
        self.assertEqual(info_state.street, Street.FLOP)

    def test_get_strategy(self):
        """Test getting a strategy for an information state."""
        info_state = InfoState(
            player_id=0,
            hole_cards=[0, 1],
            community_cards=[],
            action_history=[],
            street=Street.PREFLOP
        )

        strategy = self.drm.get_strategy(info_state)

        # Verify strategy is a valid probability distribution
        self.assertIsInstance(strategy, dict)
        total_prob = sum(strategy.values())
        self.assertAlmostEqual(total_prob, 1.0, places=7)

        # Verify all probabilities are non-negative
        self.assertTrue(all(prob >= 0 for prob in strategy.values()))

    def test_iteration_metrics(self):
        """Test the metrics returned by an iteration."""
        metrics = self.drm.iterate()

        # Check the structure of returned metrics
        expected_keys = ['iteration', 'num_info_states', 'player_0_value', 'convergence']
        for key in expected_keys:
            self.assertIn(key, metrics)

        # Verify metric types
        self.assertIsInstance(metrics['iteration'], int)
        self.assertIsInstance(metrics['num_info_states'], int)
        self.assertIsInstance(metrics['player_0_value'], float)
        self.assertIsInstance(metrics['convergence'], float)

    def test_exploitability_computation(self):
        """Test computation of exploitability."""
        # Run a few iterations to build up some strategy
        for _ in range(10):
            self.drm.iterate()

        exploitability = self.drm.compute_exploitability()

        # Exploitability should be a non-negative float
        self.assertIsInstance(exploitability, float)
        self.assertTrue(exploitability >= 0)

    def test_average_strategy(self):
        """Test computing average strategy after multiple iterations."""
        # Run multiple iterations to build strategy
        for _ in range(100):
            self.drm.iterate()

        # Create a sample information state
        info_state = InfoState(
            player_id=0,
            hole_cards=[0, 1],
            community_cards=[],
            action_history=[],
            street=Street.PREFLOP
        )

        avg_strategy = self.drm.get_average_strategy(info_state)

        # Verify average strategy properties
        self.assertIsInstance(avg_strategy, dict)
        total_prob = sum(avg_strategy.values())
        self.assertAlmostEqual(total_prob, 1.0, places=7)

        # Verify all probabilities are non-negative
        self.assertTrue(all(prob >= 0 for prob in avg_strategy.values()))


if __name__ == '__main__':
    unittest.main()