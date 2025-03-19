"""
terminology.py - Texas Hold'em terminology and explanations

This module provides standardized poker terminology, definitions, and helper
functions for consistent terminology usage throughout the PokerGTO project.
"""

from typing import Dict, List, Optional, Tuple
from enum import Enum


class TerminologyCategory(Enum):
    """Categories for poker terminology to help with organization."""
    ACTIONS = "actions"
    BETTING = "betting"
    CARDS = "cards"
    POSITIONS = "positions"
    GAME_STRUCTURE = "game_structure"
    HAND_STRENGTH = "hand_strength"
    STRATEGY = "strategy"


# Comprehensive poker terminology dictionary with explanations
POKER_TERMS = {
    "action": {
        "category": TerminologyCategory.GAME_STRUCTURE,
        "short": "Player's turn to act or betting activity",
        "definition": "Refers to the decision-making part of poker, like betting, checking, "
                      "raising, or folding. When someone says 'The action is on you,' they "
                      "mean it's your turn to make a move."
    },
    "all_in": {
        "category": TerminologyCategory.ACTIONS,
        "short": "Betting all remaining chips",
        "definition": "Betting all your remaining chips on a single hand. After going all-in, "
                      "you can't bet further, but you remain eligible to win everything you "
                      "matched in the pot."
    },
    "ante": {
        "category": TerminologyCategory.BETTING,
        "short": "Small forced bet from all players",
        "definition": "A small forced bet placed by all players at the start of certain poker "
                      "games or stages. Antes help increase pot size and encourage action."
    },
    "big_blind": {
        "category": TerminologyCategory.BETTING,
        "short": "Mandatory bet by second player left of dealer",
        "definition": "A forced bet that the second player to the left of the dealer must place "
                      "before cards are dealt. It sets the minimum bet size for the first betting "
                      "round and is usually twice the small blind."
    },
    "blinds": {
        "category": TerminologyCategory.BETTING,
        "short": "Mandatory bets before cards are dealt",
        "definition": "Mandatory bets made before cards are dealt, consisting of the small blind "
                      "(half the big blind) and big blind. Blinds rotate clockwise after each "
                      "hand to ensure fair play."
    },
    "bluff": {
        "category": TerminologyCategory.STRATEGY,
        "short": "Betting with a weak hand to deceive opponents",
        "definition": "Acting as if your cards are stronger or weaker than they really are, "
                      "in order to influence opponents' decisions. The goal is to make "
                      "opponents fold better hands or incorrectly call when you're strong."
    },
    "board": {
        "category": TerminologyCategory.CARDS,
        "short": "Community cards on the table",
        "definition": "The community cards dealt face-up on the table for all players to share, "
                      "consisting of the flop (three cards), turn (one card), and river (one card)."
    },
    "burn_card": {
        "category": TerminologyCategory.CARDS,
        "short": "Card discarded before dealing community cards",
        "definition": "A card discarded face-down by the dealer before dealing each community "
                      "card (flop, turn, river) to prevent cheating by ensuring no one sees "
                      "upcoming cards in advance."
    },
    "call": {
        "category": TerminologyCategory.ACTIONS,
        "short": "Matching the current bet",
        "definition": "Matching the amount of the current bet made by another player. Calling "
                      "allows you to continue playing the hand without raising the stakes further."
    },
    "check": {
        "category": TerminologyCategory.ACTIONS,
        "short": "Passing action without betting",
        "definition": "Choosing not to bet when no one else has bet in the current betting round, "
                      "allowing the next player to act. You can only check if there's no existing "
                      "bet to match."
    },
    "community_cards": {
        "category": TerminologyCategory.CARDS,
        "short": "Shared cards used by all players",
        "definition": "The five shared cards placed face-up in the center of the table (flop, "
                      "turn, river). Players use these cards along with their hole cards to "
                      "form the best possible five-card poker hand."
    },
    "dealer_button": {
        "category": TerminologyCategory.POSITIONS,
        "short": "Marker indicating dealer position",
        "definition": "A marker placed in front of the player acting as the dealer (in home games) "
                      "or indicating dealer position (in casinos). The button rotates clockwise "
                      "after each hand, determining who places blinds first."
    },
    "draw": {
        "category": TerminologyCategory.HAND_STRENGTH,
        "short": "Potential to complete a strong hand",
        "definition": "When you hold incomplete cards but can potentially complete a stronger hand "
                      "with future cards. Common examples include flush draws (needing one more "
                      "card of a specific suit) or straight draws (needing a particular rank to "
                      "complete a straight)."
    },
    "flop": {
        "category": TerminologyCategory.CARDS,
        "short": "First three community cards",
        "definition": "The first three community cards dealt face-up at once after the initial "
                      "betting round. Players immediately begin assessing their hands more clearly."
    },
    "fold": {
        "category": TerminologyCategory.ACTIONS,
        "short": "Discarding hand and forfeiting the pot",
        "definition": "Giving up your hand by discarding your cards instead of matching a bet or "
                      "raising. Once folded, you lose your rights to the pot for that hand."
    },
    "hole_cards": {
        "category": TerminologyCategory.CARDS,
        "short": "Player's private cards",
        "definition": "Your private two cards dealt face-down at the beginning of each hand. "
                      "These are combined with community cards to form your best five-card hand."
    },
    "kicker": {
        "category": TerminologyCategory.HAND_STRENGTH,
        "short": "Card used to break ties",
        "definition": "If two players have hands of equal strength (like pairs), the kicker is "
                      "the highest remaining card used to break ties. The player with the higher "
                      "kicker wins."
    },
    "muck": {
        "category": TerminologyCategory.ACTIONS,
        "short": "Discarding cards without showing",
        "definition": "To discard your cards without showing them after losing a hand, or the "
                      "pile of discarded cards. Mucking helps keep your strategy hidden."
    },
    "nuts": {
        "category": TerminologyCategory.HAND_STRENGTH,
        "short": "Best possible hand",
        "definition": "The absolute best possible hand given the current community cards. "
                      "Holding 'the nuts' guarantees you can't be beaten at that moment."
    },
    "outs": {
        "category": TerminologyCategory.STRATEGY,
        "short": "Cards that improve your hand",
        "definition": "The number of unseen cards remaining in the deck that will significantly "
                      "improve your current hand. For example, if you need one specific rank for "
                      "a straight, you have four outs (one for each suit)."
    },
    "pot": {
        "category": TerminologyCategory.BETTING,
        "short": "Total chips wagered in a hand",
        "definition": "The collection of all chips wagered by players during a single hand. "
                      "The player with the best hand (or the last player remaining after "
                      "everyone folds) wins the pot."
    },
    "raise": {
        "category": TerminologyCategory.ACTIONS,
        "short": "Increasing the current bet",
        "definition": "Increasing the current bet amount, forcing other players to either match "
                      "your higher bet, raise again, or fold. Raising can build the pot or "
                      "pressure opponents."
    },
    "rake": {
        "category": TerminologyCategory.GAME_STRUCTURE,
        "short": "Fee taken by casino/poker room",
        "definition": "A small fee taken by the casino or poker room from each pot as payment "
                      "for hosting the game. Online poker sites also charge rake."
    },
    "river": {
        "category": TerminologyCategory.CARDS,
        "short": "Fifth and final community card",
        "definition": "The fifth and final community card dealt face-up. After the river is "
                      "revealed, players have all information available to determine the "
                      "strength of their final hands."
    },
    "showdown": {
        "category": TerminologyCategory.GAME_STRUCTURE,
        "short": "Revealing cards to determine winner",
        "definition": "When multiple players remain after the final betting round (river), "
                      "they reveal their cards to determine who has the best five-card poker "
                      "hand and wins the pot."
    },
    "small_blind": {
        "category": TerminologyCategory.BETTING,
        "short": "Smaller forced bet left of dealer",
        "definition": "A forced bet that the player immediately to the dealer's left must place "
                      "before cards are dealt. Typically half the size of the big blind, setting "
                      "up initial pot-building action."
    },
    "split_pot": {
        "category": TerminologyCategory.GAME_STRUCTURE,
        "short": "Pot divided among tied players",
        "definition": "When two or more players have equally strong hands at showdown, the pot "
                      "is evenly divided among them."
    },
    "tilt": {
        "category": TerminologyCategory.STRATEGY,
        "short": "Playing emotionally after losses",
        "definition": "Playing poorly or emotionally after a loss, often leading to reckless "
                      "decisions. Being 'on tilt' usually results in further losses."
    },
    "turn": {
        "category": TerminologyCategory.CARDS,
        "short": "Fourth community card",
        "definition": "The fourth community card dealt face-up after the flop and before the river. "
                      "It significantly changes the strength of players' hands, making strategic "
                      "decisions important."
    },
    "under_the_gun": {
        "category": TerminologyCategory.POSITIONS,
        "short": "First player to act preflop",
        "definition": "The player who acts first before the flop, immediately to the left of the "
                      "big blind. UTG is considered a challenging position due to acting early "
                      "without seeing other players' moves."
    }
}


def get_term_definition(term: str) -> Optional[str]:
    """
    Get the full definition for a poker term.

    Args:
        term: The poker term to look up

    Returns:
        Full definition of the term, or None if term not found
    """
    normalized_term = term.lower().replace(" ", "_")
    if normalized_term in POKER_TERMS:
        return POKER_TERMS[normalized_term]["definition"]
    return None


def get_short_definition(term: str) -> Optional[str]:
    """
    Get the short definition for a poker term.

    Args:
        term: The poker term to look up

    Returns:
        Short definition of the term, or None if term not found
    """
    normalized_term = term.lower().replace(" ", "_")
    if normalized_term in POKER_TERMS:
        return POKER_TERMS[normalized_term]["short"]
    return None


def get_terms_by_category(category: TerminologyCategory) -> List[str]:
    """
    Get all terms belonging to a specific category.

    Args:
        category: The category to filter by

    Returns:
        List of terms in the specified category
    """
    return [
        term for term, data in POKER_TERMS.items()
        if data["category"] == category
    ]


def get_all_categories() -> List[TerminologyCategory]:
    """
    Get all available terminology categories.

    Returns:
        List of all terminology categories
    """
    return list(TerminologyCategory)


def create_glossary() -> Dict[str, str]:
    """
    Create a glossary mapping poker terms to their definitions.

    Returns:
        Dictionary mapping terms to definitions
    """
    return {term: data["definition"] for term, data in POKER_TERMS.items()}