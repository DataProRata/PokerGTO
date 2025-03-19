import unittest
import sys
import os
import json
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.api.slumbot import SlumbotClient
from config import SLUMBOT_API


class TestSlumbotClient(unittest.TestCase):
    def setUp(self):
        """Initialize the Slumbot client for testing."""
        self.client = SlumbotClient(
            base_url=SLUMBOT_API['base_url'],
            retries=SLUMBOT_API['retries'],
            timeout=SLUMBOT_API['timeout']
        )

    def test_card_parsing(self):
        """Test card string to index conversion."""
        test_cases = [
            ("As", 12),  # Ace of Spades
            ("Kh", 11),  # King of Hearts
            ("Qd", 10),  # Queen of Diamonds
            ("Jc", 9),  # Jack of Clubs
            ("Ts", 8),  # Ten of Spades
            ("9h", 6),  # Nine of Hearts
        ]

        for card_str, expected_index in test_cases:
            parsed_card = self.client._parse_cards([card_str])[0]
            self.assertEqual(parsed_card, expected_index,
                             f"Failed to parse {card_str}")

    def test_card_conversion(self):
        """Test conversion between card strings and indices."""
        card_indices = [0, 13, 26, 39]  # 2h, 2d, 2c, 2s
        card_strs = self.client._card_indices_to_strings(card_indices)

        self.assertEqual(card_strs, ['2h', '2d', '2c', '2s'])

        # Convert back to indices
        converted_indices = [self.client._parse_cards([s])[0] for s in card_strs]
        self.assertEqual(converted_indices, card_indices)

    def test_action_conversion(self):
        """Test converting internal actions to Slumbot API format."""
        test_cases = [
            ('FOLD', None, ('fold', None)),
            ('CHECK', None, ('check', None)),
            ('CALL', None, ('call', None)),
            ('BET', 10.0, ('bet', 10.0)),
            ('RAISE', 20.0, ('raise', 20.0)),
            ('ALL_IN', 50.0, ('raise', 50.0))
        ]

        for action, amount, expected in test_cases:
            converted = self.client.convert_action(action, amount)
            self.assertEqual(converted, expected)

    @patch('requests.request')
    def test_api_request_retry(self, mock_request):
        """Test retry mechanism for API requests."""
        # Configure mock to raise connection errors first, then succeed
        mock_request.side_effect = [
            requests.ConnectionError("First attempt failed"),
            requests.ConnectionError("Second attempt failed"),
            MagicMock(status_code=200, json=lambda: {"status": "success"})
        ]

        try:
            result = self.client._make_request('test_endpoint', method='GET')
            self.assertEqual(result, {"status": "success"})
        except Exception as e:
            self.fail(f"Retry mechanism failed: {e}")

    def test_simulated_api(self):
        """Test the simulated API functionality."""
        # Call the simulate method
        simulated_game = self.client.simulate_api()

        # Verify expected keys in the simulated game
        expected_keys = ['session', 'hand_id', 'stack', 'hole_cards', 'betting']
        for key in expected_keys:
            self.assertIn(key, simulated_game)

        # Verify hole cards
        hole_cards = simulated_game['hole_cards']
        self.assertEqual(len(hole_cards), 2)

        for card_str in hole_cards:
            # Verify card string format
            self.assertEqual(len(card_str), 2)
            self.assertIn(card_str[0], '23456789TJQKA')
            self.assertIn(card_str[1].lower(), 'hdcs')

    def test_random_card_generation(self):
        """Test random card generation methods."""
        # Generate random cards
        card_indices = self.client.get_random_cards(num_cards=5)
        card_strings = self.client._card_indices_to_strings(card_indices)

        # Verify uniqueness and proper conversion
        self.assertEqual(len(set(card_indices)), 5)  # Unique cards
        self.assertEqual(len(card_strings), 5)  # Correct number of cards

        for card_str in card_strings:
            # Verify card string format
            self.assertEqual(len(card_str), 2)
            self.assertIn(card_str[0], '23456789TJQKA')
            self.assertIn(card_str[1].lower(), 'hdcs')


if __name__ == '__main__':
    unittest.main()