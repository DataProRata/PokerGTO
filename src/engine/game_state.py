"""
game_state.py - Representation of a poker game state

This module defines the GameState class which tracks the current state of a poker hand,
including player stacks, pot, cards, and betting information.
"""

import numpy as np
from enum import Enum, auto
from typing import List, Dict, Tuple, Optional, Set


class Street(Enum):
    """Enum representing the different streets (rounds) in Texas Hold'em."""
    PREFLOP = auto()
    FLOP = auto()
    TURN = auto()
    RIVER = auto()


class Action(Enum):
    """Enum representing the possible player actions in Texas Hold'em."""
    FOLD = auto()
    CHECK = auto()
    CALL = auto()
    BET = auto()
    RAISE = auto()
    ALL_IN = auto()


class GameState:
    """
    Represents the complete state of a poker game at a specific point in time.

    This class tracks all the information needed to determine valid actions
    and evaluate expected values of different strategies.
    """

    def __init__(self, num_players: int = 2, stack_size: float = 200.0,
                 small_blind: float = 0.5, big_blind: float = 1.0):
        """
        Initialize a new poker game state.

        Args:
            num_players: Number of players (default: 2 for heads-up)
            stack_size: Starting stack size in big blinds (default: 200)
            small_blind: Small blind amount (default: 0.5)
            big_blind: Big blind amount (default: 1.0)
        """
        self.num_players = num_players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.initial_stack = stack_size * big_blind

        # Initialize game state variables
        self.reset()

    def reset(self) -> None:
        """Reset the game state to start a new hand."""
        # Player stacks
        self.stacks = [self.initial_stack] * self.num_players

        # Betting state
        self.pot = 0.0
        self.street = Street.PREFLOP
        self.current_player = 0  # Player to act (position index)
        self.button_pos = 0  # Dealer position
        self.last_aggressor = 1  # Player who made the last aggressive action

        # Betting amounts
        self.bets = [0.0] * self.num_players  # Current street bets
        self.committed = [0.0] * self.num_players  # Total committed this hand

        # Posted blinds
        self.post_blinds()

        # Action history
        self.action_history = []

        # Card state (empty until dealt)
        self.community_cards = []
        self.player_cards = [[] for _ in range(self.num_players)]

        # Player states
        self.folded = [False] * self.num_players
        self.all_in = [False] * self.num_players

    def post_blinds(self) -> None:
        """Post the small and big blinds."""
        # Small blind
        sb_pos = (self.button_pos + 1) % self.num_players
        self.stacks[sb_pos] -= self.small_blind
        self.bets[sb_pos] = self.small_blind
        self.committed[sb_pos] = self.small_blind

        # Big blind
        bb_pos = (self.button_pos + 2) % self.num_players
        self.stacks[bb_pos] -= self.big_blind
        self.bets[bb_pos] = self.big_blind
        self.committed[bb_pos] = self.big_blind

        # First to act preflop is after BB (or SB in heads-up)
        self.current_player = (self.button_pos + 3) % self.num_players
        if self.num_players == 2:
            self.current_player = self.button_pos

    def deal_hole_cards(self, cards: List[List[int]]) -> None:
        """
        Deal hole cards to players.

        Args:
            cards: List of card lists for each player (e.g., [[0, 1], [2, 3]] for 2 players)
        """
        assert len(cards) == self.num_players, "Must provide cards for each player"
        assert all(len(hand) == 2 for hand in cards), "Each player must receive 2 cards"

        self.player_cards = cards.copy()

    def deal_community_cards(self, cards: List[int], street: Street) -> None:
        """
        Deal community cards for a specific street.

        Args:
            cards: List of card indices to add to community cards
            street: The street to deal cards for (FLOP, TURN, RIVER)
        """
        self.community_cards.extend(cards)
        self.street = street

        # Reset the betting for the new street
        self.bets = [0.0] * self.num_players

        # Set first to act - after flop, it's SB (or button in heads-up)
        if street == Street.FLOP:
            self.current_player = (self.button_pos + 1) % self.num_players
            if self.num_players == 2:
                self.current_player = self.button_pos

    def apply_action(self, action: Action, amount: float = 0.0) -> bool:
        """
        Apply a player action to update the game state.

        Args:
            action: The action to apply
            amount: The bet or raise amount (if applicable)

        Returns:
            bool: True if the street is complete, False otherwise
        """
        player = self.current_player

        # Record the action in history
        self.action_history.append((player, action, amount))

        # Apply the action
        if action == Action.FOLD:
            self.folded[player] = True

        elif action == Action.CHECK:
            # No change to state, just update turn
            pass

        elif action == Action.CALL:
            # Find the maximum bet
            max_bet = max(self.bets)
            call_amount = max_bet - self.bets[player]

            # Handle all-in calls
            if call_amount > self.stacks[player]:
                call_amount = self.stacks[player]
                self.all_in[player] = True

            # Update stacks and bets
            self.stacks[player] -= call_amount
            self.bets[player] += call_amount
            self.committed[player] += call_amount

        elif action in [Action.BET, Action.RAISE]:
            # Calculate total commitment for this action
            if action == Action.BET:
                total_bet = amount
            else:  # RAISE
                max_bet = max(self.bets)
                total_bet = max_bet + amount

            # Handle all-in bets/raises
            if total_bet > self.stacks[player] + self.bets[player]:
                total_bet = self.stacks[player] + self.bets[player]
                self.all_in[player] = True

            # Calculate the additional amount the player needs to put in
            additional = total_bet - self.bets[player]

            # Update stacks and bets
            self.stacks[player] -= additional
            self.bets[player] = total_bet
            self.committed[player] += additional

            # Update last aggressor
            self.last_aggressor = player

        elif action == Action.ALL_IN:
            # Put all remaining chips into the pot
            all_in_amount = self.stacks[player]
            self.bets[player] += all_in_amount
            self.committed[player] += all_in_amount
            self.stacks[player] = 0.0
            self.all_in[player] = True

            # Update last aggressor if this is an aggressive action
            if self.bets[player] > max(b for i, b in enumerate(self.bets) if i != player):
                self.last_aggressor = player

        # Move to the next player
        return self._advance_state()

    def _advance_state(self) -> bool:
        """
        Advance to the next player or street if betting is complete.

        Returns:
            bool: True if the street is complete, False otherwise
        """
        # Find next player who hasn't folded or gone all-in
        players_in_hand = [i for i in range(self.num_players)
                           if not self.folded[i] and not self.all_in[i]]

        # If 0 or 1 players left who haven't folded, the hand is over
        if len(players_in_hand) <= 1:
            # Collect bets into the pot
            self.pot += sum(self.bets)
            self.bets = [0.0] * self.num_players
            return True

        # Check if all players have acted and bets are matched
        if all(self.bets[i] == self.bets[self.last_aggressor] or self.folded[i] or self.all_in[i]
               for i in range(self.num_players)):
            # Collect bets into the pot
            self.pot += sum(self.bets)
            self.bets = [0.0] * self.num_players

            # Street is complete
            return True

        # Find the next player to act
        start = (self.current_player + 1) % self.num_players
        for i in range(self.num_players):
            next_player = (start + i) % self.num_players
            if not self.folded[next_player] and not self.all_in[next_player]:
                self.current_player = next_player
                break

        # Street continues
        return False

    def get_legal_actions(self) -> List[Tuple[Action, float, float]]:
        """
        Get the list of legal actions for the current player.

        Returns:
            List of tuples (action, min_amount, max_amount)
            For actions without amounts (like CHECK), amounts are 0.0
        """
        # If player has folded or is all-in, no legal actions
        if self.folded[self.current_player] or self.all_in[self.current_player]:
            return []

        actions = []
        player = self.current_player
        max_bet = max(self.bets)

        # FOLD is always an option if facing a bet
        if max_bet > self.bets[player]:
            actions.append((Action.FOLD, 0.0, 0.0))

        # CHECK is an option if not facing a bet
        if max_bet == self.bets[player]:
            actions.append((Action.CHECK, 0.0, 0.0))

        # CALL is an option if facing a bet
        if max_bet > self.bets[player]:
            call_amount = max_bet - self.bets[player]
            # If can't afford the call, can only go all-in
            if call_amount >= self.stacks[player]:
                actions.append((Action.ALL_IN, self.stacks[player], self.stacks[player]))
            else:
                actions.append((Action.CALL, call_amount, call_amount))

        # BET is an option if no one has bet
        if max_bet == 0:
            # Minimum bet is 1 BB
            min_bet = self.big_blind
            max_bet = self.stacks[player]

            if min_bet <= max_bet:
                actions.append((Action.BET, min_bet, max_bet))

        # RAISE is an option if someone has bet and we can raise
        elif max_bet > 0 and self.stacks[player] > (max_bet - self.bets[player]):
            # Minimum raise is the last bet or raise amount
            # For simplicity, using 1 BB as min raise size
            min_raise = self.big_blind
            max_raise = self.stacks[player] - (max_bet - self.bets[player])

            if min_raise <= max_raise:
                actions.append((Action.RAISE, min_raise, max_raise))

        # ALL_IN is always an option if you have chips
        if self.stacks[player] > 0:
            actions.append((Action.ALL_IN, self.stacks[player], self.stacks[player]))

        return actions

    def is_terminal(self) -> bool:
        """Check if the game state is terminal (hand is over)."""
        # If only one player hasn't folded, the hand is over
        players_left = sum(1 for f in self.folded if not f)
        if players_left == 1:
            return True

        # If we've reached showdown (river betting complete)
        if self.street == Street.RIVER and sum(self.bets) == 0 and self.pot > 0:
            return True

        # If everyone is all-in or folded
        active_players = [i for i in range(self.num_players)
                          if not self.folded[i] and not self.all_in[i]]
        if not active_players and self.pot > 0:
            return True

        return False

    def get_payoffs(self, hand_strengths: Optional[List[float]] = None) -> List[float]:
        """
        Calculate payoffs for all players at a terminal state.

        Args:
            hand_strengths: Optional list of hand strength values for each player
                           (higher is better, folded hands are ignored)

        Returns:
            List of payoffs (positive for winners, negative for losers)
        """
        # Initialize payoffs as negative of the committed amounts
        payoffs = [-amt for amt in self.committed]

        # If only one player remains, they win the pot
        players_left = [i for i in range(self.num_players) if not self.folded[i]]
        if len(players_left) == 1:
            winner = players_left[0]
            payoffs[winner] += self.pot + sum(self.bets)
            return payoffs

        # For showdowns, we need hand strengths
        if hand_strengths is None:
            # If no hand strengths provided, just return the negative commitments
            return payoffs

        # Simple showdown logic - winner takes all
        # In real poker, there would be side pots for all-ins
        best_hand = -float('inf')
        winners = []

        for i in range(self.num_players):
            if not self.folded[i]:
                if hand_strengths[i] > best_hand:
                    best_hand = hand_strengths[i]
                    winners = [i]
                elif hand_strengths[i] == best_hand:
                    winners.append(i)

        # Distribute pot among winners
        pot_total = self.pot + sum(self.bets)
        for winner in winners:
            payoffs[winner] += pot_total / len(winners)

        return payoffs

    def to_features(self) -> np.ndarray:
        """
        Convert the game state to a feature vector for ML models.

        Returns:
            NumPy array with game state features
        """
        # This is a simplified example - a real implementation would be more comprehensive
        features = []

        # Basic state information
        features.append(self.street.value)
        features.append(self.pot)
        features.extend(self.stacks)
        features.extend(self.bets)
        features.extend(self.committed)

        # Player states (folded, all-in)
        features.extend([1 if f else 0 for f in self.folded])
        features.extend([1 if a else 0 for a in self.all_in])

        # Card information would need card encoding
        # This is just a placeholder
        card_features = [0] * 52  # One-hot encoding of cards

        # Set bits for community cards
        for card in self.community_cards:
            card_features[card] = 1

        # Add hole cards for current player
        if self.player_cards[self.current_player]:
            for card in self.player_cards[self.current_player]:
                card_features[card] = 1

        features.extend(card_features)

        return np.array(features, dtype=np.float32)

    def __str__(self) -> str:
        """Return a string representation of the game state."""
        state_str = f"Street: {self.street.name}\n"
        state_str += f"Pot: {self.pot}\n"
        state_str += f"Current player: {self.current_player}\n"

        for i in range(self.num_players):
            state_str += f"Player {i}: Stack={self.stacks[i]:.1f}, Bet={self.bets[i]:.1f}, "
            state_str += f"Committed={self.committed[i]:.1f}, "
            state_str += f"{'FOLDED' if self.folded[i] else 'Active'}, "
            state_str += f"{'ALL-IN' if self.all_in[i] else 'Has chips'}\n"

        return state_str