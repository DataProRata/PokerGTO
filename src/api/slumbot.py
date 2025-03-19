"""
slumbot.py - Client for interacting with the Slumbot poker AI API

This module provides a client for communicating with the Slumbot poker AI API,
allowing the system to play and learn from games against Slumbot.
"""

import requests
import logging
import time
import json
import random
from typing import Dict, List, Tuple, Optional, Any


class SlumbotClient:
    """
    Client for interacting with the Slumbot poker AI API.

    This class handles communication with the Slumbot server, including
    authentication, game creation, action submission, and results processing.
    """

    def __init__(self, base_url: str, retries: int = 3, timeout: float = 5.0,
                 verify_ssl: bool = True):
        """
        Initialize the Slumbot client.

        Args:
            base_url: Base URL for the Slumbot API
            retries: Number of times to retry failed requests (default: 3)
            timeout: Request timeout in seconds (default: 5.0)
            verify_ssl: Whether to verify SSL certificates (default: True)
        """
        self.logger = logging.getLogger(__name__)
        self.base_url = base_url.rstrip('/')
        self.retries = retries
        self.timeout = timeout
        self.verify_ssl = verify_ssl

        # Session data
        self.session = None
        self.game_id = None
        self.hand_id = None

        # Game state
        self.stack = 0.0
        self.hole_cards = []
        self.community_cards = []
        self.betting_history = []

        self.logger.info(f"Initialized Slumbot client with base URL: {base_url}")

    def _make_request(self, endpoint: str, method: str = 'GET',
                      data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make an HTTP request to the API.

        Args:
            endpoint: API endpoint to call
            method: HTTP method to use (GET, POST, etc.)
            data: Data to include in the request (for POST/PUT)

        Returns:
            API response as a dictionary
        """
        url = f"{self.base_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}

        # Retry loop
        for attempt in range(self.retries + 1):
            try:
                if method.upper() == 'GET':
                    response = requests.get(
                        url, headers=headers, timeout=self.timeout,
                        verify=self.verify_ssl, params=data
                    )
                else:  # POST, PUT, etc.
                    response = requests.request(
                        method, url, headers=headers, timeout=self.timeout,
                        verify=self.verify_ssl, json=data
                    )

                response.raise_for_status()
                return response.json()

            except requests.RequestException as e:
                self.logger.warning(f"Request to {url} failed (attempt {attempt + 1}/{self.retries + 1}): {e}")

                if attempt < self.retries:
                    # Wait before retrying (exponential backoff)
                    wait_time = 0.5 * (2 ** attempt)
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Request to {url} failed after {self.retries} retries")
                    raise

    def start_new_hand(self) -> Dict[str, Any]:
        """
        Start a new hand against Slumbot.

        Returns:
            Dictionary with information about the new hand
        """
        self.logger.info("Starting a new hand against Slumbot")

        # Reset game state
        self.hole_cards = []
        self.community_cards = []
        self.betting_history = []

        # Make API request
        try:
            response = self._make_request('new_hand', method='POST')

            # Process response
            self.session = response.get('session')
            self.game_id = response.get('game_id')
            self.hand_id = response.get('hand_id')
            self.stack = float(response.get('stack', 0.0))

            # Parse hole cards
            hole_cards_str = response.get('hole_cards', '').split()
            self.hole_cards = self._parse_cards(hole_cards_str)

            # Parse betting info
            self.betting_history = response.get('betting', [])

            self.logger.info(f"Started new hand with ID: {self.hand_id}")
            self.logger.debug(f"Hole cards: {hole_cards_str}")

            return response

        except Exception as e:
            self.logger.error(f"Failed to start new hand: {e}")
            raise

    def submit_action(self, action: str, amount: Optional[float] = None) -> Dict[str, Any]:
        """
        Submit an action to the Slumbot API.

        Args:
            action: Action to take ('fold', 'call', 'check', 'raise', 'bet')
            amount: Amount to bet or raise (if applicable)

        Returns:
            Dictionary with the result of the action
        """
        if self.session is None or self.hand_id is None:
            self.logger.error("Cannot submit action: No active hand")
            raise ValueError("No active hand")

        self.logger.info(f"Submitting action: {action}" +
                         (f" {amount}" if amount is not None else ""))

        # Prepare request data
        data = {
            'session': self.session,
            'action': action
        }

        if amount is not None and action in ['raise', 'bet']:
            data['amount'] = amount

        # Make API request
        try:
            response = self._make_request('act', method='POST', data=data)

            # Process response
            if 'betting' in response:
                self.betting_history = response['betting']

            # Update community cards if present
            if 'board' in response:
                board_str = response['board'].split()
                self.community_cards = self._parse_cards(board_str)

            # Check for hand completion
            if response.get('hand_over', False):
                self.logger.info(f"Hand {self.hand_id} completed")

                # Process the hand result
                if 'showdown' in response:
                    self.logger.info(f"Showdown result: {response['showdown']}")

                if 'winnings' in response:
                    self.logger.info(f"Winnings: {response['winnings']}")

            return response

        except Exception as e:
            self.logger.error(f"Failed to submit action: {e}")
            raise

    def _parse_cards(self, card_strings: List[str]) -> List[int]:
        """
        Parse card strings into internal card indices.

        Args:
            card_strings: List of card strings (e.g., ["As", "Kh"])

        Returns:
            List of card indices
        """
        # Map card strings to indices
        # This is a simplified version and would need to match your card representation
        rank_map = {
            '2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6,
            '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12
        }

        suit_map = {'h': 0, 'd': 1, 'c': 2, 's': 3}

        cards = []
        for card_str in card_strings:
            if len(card_str) != 2:
                self.logger.warning(f"Invalid card string: {card_str}")
                continue

            rank_char = card_str[0]
            suit_char = card_str[1].lower()

            if rank_char not in rank_map or suit_char not in suit_map:
                self.logger.warning(f"Invalid card: {card_str}")
                continue

            rank = rank_map[rank_char]
            suit = suit_map[suit_char]

            # Convert to card index (0-51)
            card_idx = rank + (suit * 13)
            cards.append(card_idx)

        return cards

    def convert_action(self, action_name: str, amount: Optional[float] = None) -> Tuple[str, Optional[float]]:
        """
        Convert internal action representation to Slumbot API format.

        Args:
            action_name: Internal action name
            amount: Action amount (if applicable)

        Returns:
            Tuple of (slumbot_action, amount)
        """
        # Map from internal action names to Slumbot API actions
        action_map = {
            'FOLD': 'fold',
            'CHECK': 'check',
            'CALL': 'call',
            'BET': 'bet',
            'RAISE': 'raise',
            'ALL_IN': 'raise'  # Slumbot doesn't have a specific "all-in" action
        }

        slumbot_action = action_map.get(action_name.upper(), action_name.lower())

        return slumbot_action, amount

    def play_hand(self, strategy_function) -> Dict[str, Any]:
        """
        Play a complete hand against Slumbot using the provided strategy function.

        Args:
            strategy_function: Function that takes the current state and returns an action

        Returns:
            Dictionary with the result of the hand
        """
        # Start a new hand
        hand_info = self.start_new_hand()

        hand_over = False
        result = None

        while not hand_over:
            # Get the current state
            state = {
                'hole_cards': self.hole_cards,
                'community_cards': self.community_cards,
                'betting_history': self.betting_history,
                'stack': self.stack
            }

            # Get the action from the strategy function
            action, amount = strategy_function(state)

            # Submit the action
            result = self.submit_action(action, amount)

            # Check if the hand is over
            hand_over = result.get('hand_over', False)

        return result

    def simulate_api(self):
        """
        Simulate the Slumbot API for testing and development.

        This method can be used when the actual API is unavailable.
        It provides a simplified simulation of the API behavior.
        """
        self.logger.warning("Using simulated Slumbot API")

        # Generate a random hand ID
        self.hand_id = f"sim_{random.randint(1000, 9999)}"
        self.session = f"sim_session_{random.randint(10000, 99999)}"
        self.stack = 200.0

        # Generate random hole cards
        available_cards = list(range(52))
        random.shuffle(available_cards)
        self.hole_cards = available_cards[:2]

        # Initialize community cards (empty at start)
        self.community_cards = []

        # Initialize betting history
        self.betting_history = []

        self.logger.info(f"Simulated new hand with ID: {self.hand_id}")

        return {
            'session': self.session,
            'hand_id': self.hand_id,
            'stack': self.stack,
            'hole_cards': self._card_indices_to_strings(self.hole_cards),
            'betting': []
        }

    def _card_indices_to_strings(self, card_indices: List[int]) -> List[str]:
        """
        Convert card indices to string representations.

        Args:
            card_indices: List of card indices (0-51)

        Returns:
            List of card strings (e.g., ["As", "Kh"])
        """
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['h', 'd', 'c', 's']

        card_strings = []
        for card_idx in card_indices:
            rank = card_idx % 13
            suit = card_idx // 13

            card_str = f"{ranks[rank]}{suits[suit]}"
            card_strings.append(card_str)

        return card_strings