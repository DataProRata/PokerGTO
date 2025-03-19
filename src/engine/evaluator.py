"""
evaluator.py - Hand evaluation for Texas Hold'em

This module provides the HandEvaluator class which evaluates and compares
poker hands. It uses a lookup-based approach for performance.
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional
import logging

# For now, we'll use a simple ranks system - in a full implementation,
# you might want to use the 'treys' library or implement a full evaluator
class HandEvaluator:
    """
    Class for evaluating and comparing poker hands.

    This is a simplified implementation for the prototype. For a complete solution,
    consider using dedicated poker hand evaluation libraries like 'treys'.
    """

    # Card representation constants
    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    SUITS = ['h', 'd', 'c', 's']  # hearts, diamonds, clubs, spades

    def __init__(self):
        """Initialize the hand evaluator with necessary lookup tables."""
        self.logger = logging.getLogger(__name__)

        # Initialize lookup tables
        self._initialize_lookups()

    def _initialize_lookups(self):
        """Initialize lookup tables for hand evaluation."""
        # This is a placeholder for the actual lookup table initialization
        # In a real implementation, this would pre-compute hand rankings

        # Map from card index (0-51) to rank (0-12) and suit (0-3)
        self.card_to_rank = {}
        self.card_to_suit = {}

        for i in range(52):
            rank = i % 13
            suit = i // 13
            self.card_to_rank[i] = rank
            self.card_to_suit[i] = suit

        # Create reverse mapping
        self.card_map = {}
        for i in range(52):
            rank = self.RANKS[i % 13]
            suit = self.SUITS[i // 13]
            card_str = rank + suit
            self.card_map[card_str] = i

    def card_to_string(self, card_idx: int) -> str:
        """
        Convert a card index to a human-readable string.

        Args:
            card_idx: Card index (0-51)

        Returns:
            String representation of the card (e.g., "Ah" for Ace of hearts)
        """
        if not 0 <= card_idx < 52:
            raise ValueError(f"Invalid card index: {card_idx}")

        rank = self.RANKS[card_idx % 13]
        suit = self.SUITS[card_idx // 13]
        return rank + suit

    def string_to_card(self, card_str: str) -> int:
        """
        Convert a card string to its index representation.

        Args:
            card_str: String representation of a card (e.g., "Ah")

        Returns:
            Card index (0-51)
        """
        if len(card_str) != 2:
            raise ValueError(f"Invalid card string: {card_str}")

        if card_str[0].upper() in "23456789":
            rank = int(card_str[0]) - 2
        elif card_str[0].upper() == "T":
            rank = 8
        elif card_str[0].upper() == "J":
            rank = 9
        elif card_str[0].upper() == "Q":
            rank = 10
        elif card_str[0].upper() == "K":
            rank = 11
        elif card_str[0].upper() == "A":
            rank = 12
        else:
            raise ValueError(f"Invalid card rank: {card_str[0]}")

        if card_str[1].lower() == "h":
            suit = 0
        elif card_str[1].lower() == "d":
            suit = 1
        elif card_str[1].lower() == "c":
            suit = 2
        elif card_str[1].lower() == "s":
            suit = 3
        else:
            raise ValueError(f"Invalid card suit: {card_str[1]}")

        return rank + suit * 13

    def evaluate_hand(self, hole_cards: List[int], community_cards: List[int]) -> int:
        """
        Evaluate a poker hand and return its strength.

        Args:
            hole_cards: List of hole card indices (0-51)
            community_cards: List of community card indices (0-51)

        Returns:
            Hand strength value (higher is better)
        """
        # Combine hole cards and community cards
        all_cards = hole_cards + community_cards

        # Check for valid input
        if len(all_cards) < 5:
            raise ValueError("Need at least 5 cards to evaluate a hand")

        # Count ranks and suits
        rank_counts = [0] * 13
        suit_counts = [0] * 4

        for card in all_cards:
            rank = self.card_to_rank[card]
            suit = self.card_to_suit[card]
            rank_counts[rank] += 1
            suit_counts[suit] += 1

        # Check for flush
        flush_suit = -1
        for suit, count in enumerate(suit_counts):
            if count >= 5:
                flush_suit = suit
                break

        # Check for straight
        straight = False
        straight_high = -1

        # Handle Ace as both high and low
        if rank_counts[12] > 0:  # If we have an Ace
            rank_counts.insert(0, rank_counts[12])
        else:
            rank_counts.insert(0, 0)

        # Look for 5 consecutive ranks
        for i in range(len(rank_counts) - 4):
            if all(rank_counts[i + j] > 0 for j in range(5)):
                straight = True
                straight_high = i + 4
                break

        # Remove the duplicated Ace
        rank_counts.pop(0)

        # Find the highest card in a flush
        flush_cards = []
        if flush_suit >= 0:
            for card in all_cards:
                if self.card_to_suit[card] == flush_suit:
                    flush_cards.append(self.card_to_rank[card])

        # Determine hand type and strength
        # Simplified hand ranking for prototype
        # Hand type multipliers - higher types have higher multipliers
        hand_multipliers = {
            "high_card": 1,
            "pair": 100,
            "two_pair": 10000,
            "three_of_kind": 1000000,
            "straight": 100000000,
            "flush": 10000000000,
            "full_house": 1000000000000,
            "four_of_kind": 100000000000000,
            "straight_flush": 10000000000000000
        }

        # Detect hand type
        # Straight flush
        if straight and flush_suit >= 0 and self._is_straight_flush(all_cards, flush_suit):
            # Find the highest card in the straight flush
            sf_high = self._get_straight_flush_high(all_cards, flush_suit)
            return hand_multipliers["straight_flush"] + sf_high

        # Four of a kind
        four_rank = -1
        for rank, count in enumerate(rank_counts):
            if count >= 4:
                four_rank = rank
                break

        if four_rank >= 0:
            # Find the highest remaining card
            kicker = max(rank for rank, count in enumerate(rank_counts) if rank != four_rank and count > 0)
            return hand_multipliers["four_of_kind"] + four_rank * 100 + kicker

        # Full house
        three_rank = -1
        for rank, count in enumerate(rank_counts):
            if count >= 3:
                three_rank = rank
                break

        pair_rank = -1
        if three_rank >= 0:
            for rank, count in enumerate(rank_counts):
                if count >= 2 and rank != three_rank:
                    pair_rank = max(pair_rank, rank)

        if three_rank >= 0 and pair_rank >= 0:
            return hand_multipliers["full_house"] + three_rank * 100 + pair_rank

        # Flush
        if flush_suit >= 0:
            # Sort flush cards in descending order
            flush_cards.sort(reverse=True)
            # Take the 5 highest cards to rank the flush
            value = 0
            for i, rank in enumerate(flush_cards[:5]):
                value += rank * (100 ** (4 - i))
            return hand_multipliers["flush"] + value

        # Straight
        if straight:
            return hand_multipliers["straight"] + straight_high

        # Three of a kind
        if three_rank >= 0:
            # Find the two highest remaining cards
            kickers = sorted([rank for rank, count in enumerate(rank_counts)
                              if rank != three_rank and count > 0], reverse=True)
            return (hand_multipliers["three_of_kind"] + three_rank * 10000 +
                    kickers[0] * 100 + kickers[1])

        # Two pair
        pairs = [rank for rank, count in enumerate(rank_counts) if count >= 2]
        if len(pairs) >= 2:
            # Sort pairs in descending order
            pairs.sort(reverse=True)
            # Find the highest remaining card
            kicker = max([rank for rank, count in enumerate(rank_counts)
                          if rank not in pairs[:2] and count > 0], default=0)
            return (hand_multipliers["two_pair"] + pairs[0] * 10000 +
                    pairs[1] * 100 + kicker)

        # Pair
        if len(pairs) == 1:
            # Find the three highest remaining cards
            kickers = sorted([rank for rank, count in enumerate(rank_counts)
                              if rank != pairs[0] and count > 0], reverse=True)
            return (hand_multipliers["pair"] + pairs[0] * 1000000 +
                    kickers[0] * 10000 + kickers[1] * 100 + kickers[2])

        # High card
        # Sort ranks in descending order
        ranks = sorted([rank for rank, count in enumerate(rank_counts) if count > 0], reverse=True)
        value = 0
        for i, rank in enumerate(ranks[:5]):
            value += rank * (100 ** (4 - i))
        return hand_multipliers["high_card"] + value

    def _is_straight_flush(self, cards: List[int], suit: int) -> bool:
        """
        Check if there is a straight flush in the given suit.

        Args:
            cards: List of card indices
            suit: Suit to check for straight flush

        Returns:
            True if there is a straight flush, False otherwise
        """
        # Get all ranks of the specified suit
        suit_ranks = []
        for card in cards:
            if self.card_to_suit[card] == suit:
                suit_ranks.append(self.card_to_rank[card])

        # Need at least 5 cards of the same suit for a straight flush
        if len(suit_ranks) < 5:
            return False

        # Convert to a set for faster lookups
        rank_set = set(suit_ranks)

        # Handle Ace as both high and low
        if 12 in rank_set:  # If we have an Ace
            rank_set.add(-1)  # Add a low Ace

        # Check for 5 consecutive ranks
        for i in range(-1, 9):  # -1 (low Ace) to 8 (9), allowing Ace through King
            if all((i + j) in rank_set for j in range(5)):
                return True

        return False

    def _get_straight_flush_high(self, cards: List[int], suit: int) -> int:
        """
        Get the highest card in a straight flush.

        Args:
            cards: List of card indices
            suit: Suit of the straight flush

        Returns:
            Highest rank in the straight flush
        """
        # Get all ranks of the specified suit
        suit_ranks = []
        for card in cards:
            if self.card_to_suit[card] == suit:
                suit_ranks.append(self.card_to_rank[card])

        # Convert to a set for faster lookups
        rank_set = set(suit_ranks)

        # Handle Ace as both high and low
        if 12 in rank_set:  # If we have an Ace
            rank_set.add(-1)  # Add a low Ace

        # Find the highest straight flush
        for i in range(8, -2, -1):  # 8 (9) down to -1 (low Ace)
            if all((i + j) in rank_set for j in range(5)):
                if i == -1:  # Ace-to-5 straight
                    return 3  # 5-high straight flush
                return i + 4  # Highest card in the straight flush

        # Shouldn't reach here if _is_straight_flush was called first
        return -1

    def compare_hands(self, hole_cards1: List[int], hole_cards2: List[int],
                     community_cards: List[int]) -> int:
        """
        Compare two poker hands and determine the winner.

        Args:
            hole_cards1: First player's hole cards
            hole_cards2: Second player's hole cards
            community_cards: Community cards

        Returns:
            1 if hand1 wins, -1 if hand2 wins, 0 for a tie
        """
        strength1 = self.evaluate_hand(hole_cards1, community_cards)
        strength2 = self.evaluate_hand(hole_cards2, community_cards)

        if strength1 > strength2:
            return 1
        elif strength1 < strength2:
            return -1
        else:
            return 0

    def get_hand_type(self, hand_value: int) -> str:
        """
        Get a string representation of the hand type.

        Args:
            hand_value: Hand strength value from evaluate_hand

        Returns:
            String representing the hand type (e.g., "Flush", "Full House")
        """
        # Determine hand type based on the magnitude of the hand value
        if hand_value >= 10000000000000000:
            return "Straight Flush"
        elif hand_value >= 100000000000000:
            return "Four of a Kind"
        elif hand_value >= 1000000000000:
            return "Full House"
        elif hand_value >= 10000000000:
            return "Flush"
        elif hand_value >= 100000000:
            return "Straight"
        elif hand_value >= 1000000:
            return "Three of a Kind"
        elif hand_value >= 10000:
            return "Two Pair"
        elif hand_value >= 100:
            return "Pair"
        else:
            return "High Card"

    def get_hand_description(self, hole_cards: List[int], community_cards: List[int]) -> str:
        """
        Get a human-readable description of a poker hand.

        Args:
            hole_cards: Player's hole cards
            community_cards: Community cards

        Returns:
            String describing the hand (e.g., "Flush, Ace high")
        """
        hand_value = self.evaluate_hand(hole_cards, community_cards)
        hand_type = self.get_hand_type(hand_value)

        # This is a simplified description
        # A complete implementation would extract the specific cards that make up the hand
        hole_card_strs = [self.card_to_string(card) for card in hole_cards]
        community_card_strs = [self.card_to_string(card) for card in community_cards]

        return f"{hand_type} ({', '.join(hole_card_strs)} | {', '.join(community_card_strs)})"