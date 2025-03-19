"""
tooltips.py - Tooltips and help information for the PokerGTO interface

This module provides tooltips, help text, and explanations for the interactive
poker interface, drawing from standard poker terminology.
"""

import logging
from typing import Dict, List, Optional, Tuple
from src.engine.game_state import Street, Action
from src.utils.terminology import get_term_definition, get_short_definition


class TooltipGenerator:
    """
    Generates contextual tooltips and help information for poker players.

    This class provides explanations of poker concepts, actions, and strategic advice
    based on the current game state and user settings.
    """

    def __init__(self, detail_level: str = "medium"):
        """
        Initialize the tooltip generator.

        Args:
            detail_level: Level of detail for tooltips ("low", "medium", "high")
        """
        self.logger = logging.getLogger(__name__)
        self.detail_level = detail_level.lower()

        # Validate detail level
        if self.detail_level not in ["low", "medium", "high"]:
            self.logger.warning(f"Invalid detail level '{detail_level}'. Using 'medium' instead.")
            self.detail_level = "medium"

    def get_action_tooltip(self, action: Action) -> str:
        """
        Get tooltip for a specific poker action.

        Args:
            action: The poker action

        Returns:
            Tooltip text explaining the action
        """
        action_tooltips = {
            Action.FOLD: get_term_definition("fold"),
            Action.CHECK: get_term_definition("check"),
            Action.CALL: get_term_definition("call"),
            Action.BET: get_term_definition("bet") or "Place chips into the pot to start the betting.",
            Action.RAISE: get_term_definition("raise"),
            Action.ALL_IN: get_term_definition("all_in")
        }

        return action_tooltips.get(action, "Make a poker action.")

    def get_street_tooltip(self, street: Street) -> str:
        """
        Get tooltip for a specific poker street.

        Args:
            street: The poker street/round

        Returns:
            Tooltip text explaining the street
        """
        street_tooltips = {
            Street.PREFLOP: "The initial betting round before any community cards are dealt.",
            Street.FLOP: get_term_definition("flop"),
            Street.TURN: get_term_definition("turn"),
            Street.RIVER: get_term_definition("river")
        }

        return street_tooltips.get(street, "A betting round in poker.")

    def get_position_tooltip(self, position: str) -> str:
        """
        Get tooltip for a poker table position.

        Args:
            position: Position name (e.g., "Button", "Small Blind", "UTG")

        Returns:
            Tooltip text explaining the position
        """
        position_tooltips = {
            "Button": get_term_definition("dealer_button"),
            "Small Blind": get_term_definition("small_blind"),
            "Big Blind": get_term_definition("big_blind"),
            "UTG": get_term_definition("under_the_gun"),
            "Middle Position": "Position in the middle of the table, offering a balanced view of players yet to act.",
            "Cutoff": "The position to the right of the button, a good position for stealing blinds.",
            "Hijack": "Position to the right of the cutoff, offering good position with many players yet to act."
        }

        return position_tooltips.get(position, "A position at the poker table.")

    def get_hand_strength_tooltip(self, hand_type: str) -> str:
        """
        Get tooltip for a poker hand type.

        Args:
            hand_type: The type of poker hand (e.g., "Flush", "Full House")

        Returns:
            Tooltip text explaining the hand type
        """
        hand_tooltips = {
            "High Card": "A hand with no matching cards, valued by its highest card.",
            "Pair": "Two cards of the same rank, with three other unmatched cards.",
            "Two Pair": "Two different pairs, with one unmatched card.",
            "Three of a Kind": "Three cards of the same rank, with two unmatched cards.",
            "Straight": "Five consecutive cards of any suit.",
            "Flush": "Five cards of the same suit, not in sequence.",
            "Full House": "Three cards of one rank and two of another rank.",
            "Four of a Kind": "Four cards of the same rank, with one unmatched card.",
            "Straight Flush": "Five consecutive cards of the same suit.",
            "Royal Flush": "A straight flush from Ten to Ace, the highest possible hand."
        }

        return hand_tooltips.get(hand_type, "A poker hand.")

    def get_strategic_tip(self, street: Street, hand_strength: float, position: str) -> str:
        """
        Get a strategic tip based on the current game state.

        Args:
            street: Current street/round
            hand_strength: Estimated hand strength (0.0-1.0)
            position: Player's position

        Returns:
            Strategic advice for the current situation
        """
        # Only provide detailed strategy tips at high detail level
        if self.detail_level != "high":
            return None

        # Simple strategy matrix based on street, position, and hand strength
        if hand_strength > 0.8:  # Very strong hand
            if street == Street.PREFLOP:
                return "With a premium hand, consider raising 3-4x the big blind."
            elif street == Street.FLOP:
                return "With a strong hand on the flop, build the pot with a substantial bet."
            elif street == Street.TURN or street == Street.RIVER:
                return "With a very strong hand on late streets, consider whether a bet or check-raise will maximize value."

        elif 0.6 <= hand_strength <= 0.8:  # Strong hand
            if position == "Button" or position == "Cutoff":
                return "In late position with a strong hand, you can often raise to isolate weaker players."
            else:
                return "With a strong hand, be willing to play for a significant portion of your stack."

        elif 0.4 <= hand_strength < 0.6:  # Medium hand
            if street == Street.PREFLOP:
                if position in ["Button", "Cutoff", "Hijack"]:
                    return "Medium hands play better in position. Consider raising or calling depending on earlier action."
                else:
                    return "Be cautious with medium-strength hands in early position."
            else:
                return "Medium-strength hands should be played cautiously. Consider pot odds and your opponents' likely ranges."

        else:  # Weak hand
            if position == "Button" and street == Street.PREFLOP:
                return "The button is the best position. Even with weaker hands, consider stealing the blinds if no one has entered the pot."
            else:
                return "With a weak hand, be prepared to fold to aggression unless you have great pot odds or a strong draw."

        return "Consider your hand strength, position, and opponents' tendencies when deciding your action."

    def get_outs_explanation(self, outs: int) -> str:
        """
        Get an explanation of drawing odds based on number of outs.

        Args:
            outs: Number of cards that will improve the hand

        Returns:
            Explanation of drawing odds
        """
        if outs <= 0:
            return "With no outs, your hand cannot improve."

        # Rule of 2 and 4 for one or two cards to come
        turn_odds = outs * 2
        river_odds = outs * 4

        if self.detail_level == "low":
            return f"With {outs} outs, you have roughly a {turn_odds}% chance to hit by the next card."
        else:
            return (
                f"With {outs} outs, you have approximately:\n"
                f"- {turn_odds}% chance to hit on the next card\n"
                f"- {river_odds}% chance to hit by the river (seeing both cards)"
            )

    def get_pot_odds_advice(self, call_amount: float, pot_size: float) -> str:
        """
        Get advice about pot odds for a potential call.

        Args:
            call_amount: Amount needed to call
            pot_size: Current size of the pot

        Returns:
            Advice about pot odds
        """
        if call_amount <= 0:
            return "No call necessary - you can check."

        # Calculate pot odds
        pot_odds = call_amount / (pot_size + call_amount)
        pot_odds_percent = pot_odds * 100

        if self.detail_level == "low":
            return f"You need to call {call_amount} into a pot of {pot_size} (roughly {pot_odds_percent:.1f}% pot odds)."
        else:
            return (
                f"Pot odds: {pot_odds_percent:.1f}%\n"
                f"You need to call {call_amount} into a pot of {pot_size}.\n"
                f"Your hand needs to win more than {pot_odds_percent:.1f}% of the time to make this call profitable."
            )

    def get_concept_explanation(self, concept: str) -> str:
        """
        Get explanation of a poker concept.

        Args:
            concept: Name of the poker concept

        Returns:
            Explanation of the concept
        """
        # Try to get from terminology module first
        definition = get_term_definition(concept)
        if definition:
            return definition

        # Fallback for concepts not in the terminology module
        concept_explanations = {
            "GTO": "Game Theory Optimal (GTO) refers to a perfect strategy that cannot be exploited, "
                   "regardless of how your opponent plays. It's a balanced approach that doesn't rely "
                   "on exploiting specific opponents.",

            "Equity": "Your share of the pot based on your chances of winning. For example, if you have "
                      "a 30% chance of winning a $100 pot, your equity is $30.",

            "Range": "The set of all possible hands a player could have in a specific situation, based "
                     "on their actions. Thinking in terms of ranges rather than specific hands is crucial "
                     "for advanced poker strategy.",

            "C-bet": "Continuation bet - a bet made on the flop by the player who was the aggressor pre-flop. "
                     "It's often used whether you hit the flop or not, as a way to continue the aggression."
        }

        return concept_explanations.get(concept, f"No explanation available for '{concept}'.")

    def get_help_tip(self) -> str:
        """
        Get a random helpful poker tip.

        Returns:
            A helpful poker tip
        """
        import random

        tips = [
            "Position is power in poker. The later your position, the more information you have when acting.",
            "Pay attention to bet sizing. Larger bets often indicate stronger hands.",
            "Your table image matters. If you've been playing tight, your bets are more likely to be respected.",
            "Don't play too many hands. Being selective about starting hands is key to long-term success.",
            "Observe your opponents carefully. Look for patterns and tendencies in their play.",
            "Remember that poker is a long-term game. Don't let short-term results affect your decision-making.",
            "Calculate pot odds before calling. Make sure the price you're paying matches your chances of winning.",
            "Bluffing should be strategic, not random. Select good spots against the right opponents.",
            "Playing suited cards gives you flush possibilities, but don't overvalue small suited cards.",
            "Adjust your play based on stack sizes. Different stack-to-pot ratios call for different strategies."
        ]

        return random.choice(tips)