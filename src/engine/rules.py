"""
rules.py - Texas Hold'em rules and gameplay logic

This module defines the rules and constraints of Texas Hold'em poker,
including betting structures, hand progression, and valid actions.
"""

from typing import List, Dict, Tuple, Optional, Set
import random
import numpy as np


class PokerRules:
    """
    Encapsulates the rules of Texas Hold'em poker.

    This class defines the game constraints, betting rules,
    and handles deck management and card dealing.
    """

    def __init__(self, small_blind: float = 0.5, big_blind: float = 1.0,
                 stack_size: float = 200.0, max_raises: int = 4):
        """
        Initialize poker rules with specified parameters.

        Args:
            small_blind: Small blind amount (default: 0.5)
            big_blind: Big blind amount (default: 1.0)
            stack_size: Starting stack size in big blinds (default: 200)
            max_raises: Maximum number of raises per betting round (default: 4)
        """
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.stack_size = stack_size
        self.max_raises = max_raises

        # Initialize the deck of cards
        self.initialize_deck()

    def initialize_deck(self) -> None:
        """Initialize a standard 52-card deck."""
        self.deck = list(range(52))
        self.current_deck = self.deck.copy()

    def shuffle_deck(self) -> None:
        """Shuffle the current deck of cards."""
        self.current_deck = self.deck.copy()
        random.shuffle(self.current_deck)

    def deal_cards(self, num_cards: int) -> List[int]:
        """
        Deal a specified number of cards from the deck.

        Args:
            num_cards: Number of cards to deal

        Returns:
            List of dealt card indices
        """
        if num_cards > len(self.current_deck):
            raise ValueError(f"Cannot deal {num_cards} cards from a deck with {len(self.current_deck)} cards.")

        dealt_cards = self.current_deck[:num_cards]
        self.current_deck = self.current_deck[num_cards:]

        return dealt_cards

    def deal_hole_cards(self, num_players: int) -> List[List[int]]:
        """
        Deal hole cards to all players.

        Args:
            num_players: Number of players in the game

        Returns:
            List of hole card pairs for each player
        """
        self.shuffle_deck()
        hole_cards = []

        for _ in range(num_players):
            player_cards = self.deal_cards(2)
            hole_cards.append(player_cards)

        return hole_cards

    def deal_flop(self) -> List[int]:
        """
        Deal the flop (first three community cards).

        Returns:
            List of three card indices for the flop
        """
        # Burn a card first (standard poker procedure)
        self.deal_cards(1)

        # Deal three cards for the flop
        return self.deal_cards(3)

    def deal_turn(self) -> List[int]:
        """
        Deal the turn (fourth community card).

        Returns:
            List containing the turn card index
        """
        # Burn a card first
        self.deal_cards(1)

        # Deal one card for the turn
        return self.deal_cards(1)

    def deal_river(self) -> List[int]:
        """
        Deal the river (fifth community card).

        Returns:
            List containing the river card index
        """
        # Burn a card first
        self.deal_cards(1)

        # Deal one card for the river
        return self.deal_cards(1)

    def get_min_bet(self, street_num: int) -> float:
        """
        Get the minimum bet size for a given street.

        Args:
            street_num: Street number (0=preflop, 1=flop, 2=turn, 3=river)

        Returns:
            Minimum bet size (in big blinds)
        """
        return self.big_blind

    def get_min_raise(self, current_bet: float, street_num: int) -> float:
        """
        Get the minimum raise size for a given street and current bet.

        Args:
            current_bet: Current bet size to raise over
            street_num: Street number (0=preflop, 1=flop, 2=turn, 3=river)

        Returns:
            Minimum raise size (in big blinds)
        """
        # Standard Texas Hold'em rules: min raise is the size of the previous bet/raise
        # For simplicity, we're using 1 BB as the minimum raise size
        return current_bet + self.big_blind

    def is_valid_bet(self, amount: float, player_stack: float, min_bet: float) -> bool:
        """
        Check if a bet amount is valid.

        Args:
            amount: Proposed bet amount
            player_stack: Player's remaining stack
            min_bet: Minimum allowed bet amount

        Returns:
            True if the bet is valid, False otherwise
        """
        # Bet must be at least the minimum and no more than the player's stack
        return min_bet <= amount <= player_stack

    def is_valid_raise(self, amount: float, player_stack: float,
                       min_raise: float, current_bet: float) -> bool:
        """
        Check if a raise amount is valid.

        Args:
            amount: Proposed raise amount
            player_stack: Player's remaining stack
            min_raise: Minimum allowed raise amount
            current_bet: Current bet amount

        Returns:
            True if the raise is valid, False otherwise
        """
        # Raise must be at least the minimum and no more than the player's stack
        # The total amount to call after raising would be current_bet + amount
        return (min_raise <= amount and
                current_bet + amount <= player_stack)

    def get_betting_rounds(self) -> List[str]:
        """
        Get the list of betting rounds in a poker hand.

        Returns:
            List of betting round names
        """
        return ["preflop", "flop", "turn", "river"]

    def get_num_betting_rounds(self) -> int:
        """
        Get the number of betting rounds in a poker hand.

        Returns:
            Number of betting rounds
        """
        return len(self.get_betting_rounds())

    def get_random_cards(self, num_cards: int, excluded_cards: List[int] = None) -> List[int]:
        """
        Generate random card indices, excluding any specified cards.

        Args:
            num_cards: Number of random cards to generate
            excluded_cards: List of card indices to exclude (default: None)

        Returns:
            List of random card indices
        """
        if excluded_cards is None:
            excluded_cards = []

        available_cards = [card for card in self.deck if card not in excluded_cards]

        if num_cards > len(available_cards):
            raise ValueError(f"Cannot deal {num_cards} cards with {len(excluded_cards)} already excluded.")

        return random.sample(available_cards, num_cards)

    def create_random_game(self, num_players: int) -> Tuple[List[List[int]], List[int]]:
        """
        Create a random poker game scenario.

        Args:
            num_players: Number of players in the game

        Returns:
            Tuple of (hole_cards, community_cards) where:
                hole_cards: List of hole card pairs for each player
                community_cards: List of community cards (flop, turn, river)
        """
        self.shuffle_deck()

        # Deal hole cards to each player
        hole_cards = []
        for _ in range(num_players):
            player_cards = self.deal_cards(2)
            hole_cards.append(player_cards)

        # Deal community cards
        # Burn a card before the flop
        self.deal_cards(1)
        flop = self.deal_cards(3)

        # Burn a card before the turn
        self.deal_cards(1)
        turn = self.deal_cards(1)

        # Burn a card before the river
        self.deal_cards(1)
        river = self.deal_cards(1)

        community_cards = flop + turn + river

        return hole_cards, community_cards

    def is_game_over(self, folded: List[bool], all_in: List[bool], community_cards: List[int]) -> bool:
        """
        Check if the poker hand is over.

        Args:
            folded: List indicating which players have folded
            all_in: List indicating which players are all-in
            community_cards: List of community cards

        Returns:
            True if the hand is over, False otherwise
        """
        # Only one player left (all others folded)
        players_left = sum(1 for f in folded if not f)
        if players_left <= 1:
            return True

        # All players are either all-in or folded
        active_players = sum(1 for i, f in enumerate(folded) if not f and not all_in[i])
        if active_players == 0:
            return True

        # River betting is complete (all 5 community cards are dealt)
        if len(community_cards) == 5:
            return active_players == 0 or all(all_in[i] or folded[i] for i in range(len(folded)))

        return False