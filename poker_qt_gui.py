
# !/usr/bin/env python
"""
poker_qt_gui.py - Elegant Qt6-based poker game with machine learning strategies

A sophisticated poker game interface combining advanced AI with beautiful design.
"""

import os
import sys
import random
import numpy as np
import torch
from pathlib import Path

# Add the project directory to the path
project_root = Path(os.path.dirname(os.path.abspath(__file__)))
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Qt imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QGraphicsDropShadowEffect,
    QStackedWidget, QDialog, QSpinBox, QDialogButtonBox, QMessageBox
)
from PyQt6.QtGui import (
    QFont, QColor, QPainter, QPen, QLinearGradient,
    QRadialGradient, QGradient, QBrush, QIcon
)
from PyQt6.QtCore import Qt, QSize, QPropertyAnimation, QEasingCurve

# Poker project imports
from src.engine.game_state import GameState, Street, Action
from src.engine.evaluator import HandEvaluator
from src.engine.rules import PokerRules
from src.algorithms.discounted_regret import DiscountedRegretMinimization, InfoState
import config


class StyledButton(QPushButton):
    """Custom styled button with hover and click effects."""

    def __init__(self, text, color="#4CAF50", text_color="white"):
        super().__init__(text)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: {text_color};
                border: none;
                padding: 10px 20px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
                transition: all 0.3s ease;
            }}
            QPushButton:hover {{
                background-color: {self._darken_color(color)};
            }}
            QPushButton:pressed {{
                background-color: {self._darken_color(color, 0.7)};
                transform: scale(0.95);
            }}
        """)

        # Add shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(3, 3)
        self.setGraphicsEffect(shadow)

    def _darken_color(self, hex_color, factor=0.8):
        """Darken a hex color."""
        r = int(int(hex_color[1:3], 16) * factor)
        g = int(int(hex_color[3:5], 16) * factor)
        b = int(int(hex_color[5:7], 16) * factor)
        return f'#{r:02x}{g:02x}{b:02x}'


class CardWidget(QWidget):
    """Elegant card rendering widget."""

    def __init__(self, card_index=None, parent=None):
        super().__init__(parent)
        self.card_index = card_index
        self.setFixedSize(100, 150)
        self.setStyleSheet("background:transparent;")

    def paintEvent(self, event):
        """Custom card painting method."""
        if self.card_index is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Card background gradient
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(240, 240, 240))
        gradient.setColorAt(1, QColor(220, 220, 220))

        # Card base
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(QColor(100, 100, 100), 2, Qt.PenStyle.SolidLine))
        painter.drawRoundedRect(0, 0, self.width(), self.height(), 10, 10)

        # Convert card to readable format
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['♥', '♦', '♣', '♠']

        rank_idx = self.card_index % 13
        suit_idx = self.card_index // 13

        rank = ranks[rank_idx]
        suit = suits[suit_idx]

        # Determine color based on suit
        suit_color = (Qt.GlobalColor.red if suit_idx in [0, 1]
                      else Qt.GlobalColor.black)

        # Draw rank
        painter.setPen(QPen(suit_color))
        painter.setFont(QFont('Arial', 36, QFont.Weight.Bold))
        painter.drawText(
            self.width() // 2 - 25,
            self.height() // 2 - 20,
            rank
        )

        # Draw suit
        painter.setFont(QFont('Arial', 24))
        painter.drawText(
            self.width() // 2 - 20,
            self.height() // 2 + 40,
            suit
        )


class PokerTableWidget(QWidget):
    """Advanced poker game widget with elegant design."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Game initialization
        self.rules = PokerRules(
            small_blind=config.GAME_PARAMS['small_blind'],
            big_blind=config.GAME_PARAMS['big_blind'],
            stack_size=config.GAME_PARAMS['stack_size'],
            max_raises=config.GAME_PARAMS['max_raises_per_street']
        )
        self.evaluator = HandEvaluator()

        # Load DRM model
        self.drm = self._load_model()

        # Game state tracking
        self.game_state = None
        self.hole_cards = None
        self.player_wins = 0
        self.ai_wins = 0

        # Set up elegant styling
        self.setup_ui()

        # Start first hand
        self.start_new_hand()

    def setup_ui(self):
        """Create an elegant, modern poker table interface."""
        # Main layout
        main_layout = QVBoxLayout()

        # Top section with game info
        top_section = QHBoxLayout()

        # Game status
        self.status_label = QLabel("PokerGTO: New Game")
        self.status_label.setFont(QFont('Arial', 16, QFont.Weight.Bold))
        self.status_label.setStyleSheet("color: #333; background-color: rgba(255,255,255,0.7); padding: 5px;")

        # Scoreboard
        self.scoreboard_label = QLabel("Score - You: 0, AI: 0")
        self.scoreboard_label.setFont(QFont('Arial', 14))
        self.scoreboard_label.setStyleSheet("color: #555; background-color: rgba(255,255,255,0.7); padding: 5px;")

        top_section.addWidget(self.status_label)
        top_section.addStretch()
        top_section.addWidget(self.scoreboard_label)

        main_layout.addLayout(top_section)

        # Card display area
        card_display_layout = QHBoxLayout()

        # Player hand cards
        self.player_cards_layout = QHBoxLayout()
        player_cards_widget = QWidget()
        player_cards_widget.setLayout(self.player_cards_layout)
        player_cards_widget.setStyleSheet("background-color: rgba(0,0,0,0.1); border-radius: 10px;")

        # Community cards
        self.community_cards_layout = QHBoxLayout()
        community_cards_widget = QWidget()
        community_cards_widget.setLayout(self.community_cards_layout)
        community_cards_widget.setStyleSheet("background-color: rgba(0,0,0,0.1); border-radius: 10px;")

        # Add card sections to main layout
        card_display_layout.addWidget(player_cards_widget)
        card_display_layout.addWidget(community_cards_widget)

        main_layout.addLayout(card_display_layout)

        # Action buttons
        action_layout = QHBoxLayout()
        button_colors = [
            ("#2196F3", "white"),  # Check - blue
            ("#4CAF50", "white"),  # Bet - green
            ("#F44336", "white"),  # Fold - red
            ("#9C27B0", "white")  # All-In - purple
        ]

        action_buttons = [
            ("Check", self.player_check),
            ("Bet", self.player_bet),
            ("Fold", self.player_fold),
            ("All In", self.player_all_in)
        ]

        for (text, action), (color, text_color) in zip(action_buttons, button_colors):
            btn = StyledButton(text, color, text_color)
            btn.clicked.connect(action)
            action_layout.addWidget(btn)

        main_layout.addLayout(action_layout)

        # Log area
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("""
            QTextEdit {
                background-color: rgba(255,255,255,0.7);
                border-radius: 10px;
                padding: 10px;
                font-size: 12px;
            }
        """)

        main_layout.addWidget(self.log_area)

        # Set background gradient
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #4e9525, 
                    stop:1 #2c5e17
                );
                color: white;
            }
        """)

        self.setLayout(main_layout)

    def _load_model(self):
        """Load the trained DRM model."""
        drm = DiscountedRegretMinimization(
            rules=self.rules,
            evaluator=self.evaluator,
            discount_factor=config.DRM_PARAMS['discount_factor']
        )

        model_path = config.ADAPTIVE_MODEL_PATH
        try:
            drm.load_model(model_path)
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Could not load model: {e}")

        return drm

    def start_new_hand(self):
        """Initialize a new poker hand."""
        # Reset game state
        self.game_state = GameState(
            num_players=2,
            stack_size=config.GAME_PARAMS['stack_size'],
            small_blind=config.GAME_PARAMS['small_blind'],
            big_blind=config.GAME_PARAMS['big_blind']
        )

        # Deal hole cards
        self.hole_cards = self.rules.deal_hole_cards(num_players=2)
        self.game_state.deal_hole_cards(self.hole_cards)

        # Clear previous card displays
        self._clear_card_layouts()

        # Display player's hand
        player_cards = self.hole_cards[0]
        for card in player_cards:
            card_widget = CardWidget(card)
            self.player_cards_layout.addWidget(card_widget)

        # Update UI elements
        self.status_label.setText(f"Street: {self.game_state.street.name}")

        # Log the start of a new hand
        self.log(f"New hand started. Your hand: {self._cards_to_string(player_cards)}")

    def _clear_card_layouts(self):
        """Clear all card layouts."""
        # Clear player hand
        while self.player_cards_layout.count():
            child = self.player_cards_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Clear community cards
        while self.community_cards_layout.count():
            child = self.community_cards_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def player_check(self):
        """Handle player's check action."""
        self.player_action(Action.CHECK)

    def player_bet(self):
        """Open bet dialog and handle betting."""
        # Determine bet range based on game state
        legal_actions = self.game_state.get_legal_actions()

        # Find bet/raise actions
        bet_actions = [
            (action, min_amount, max_amount)
            for action, min_amount, max_amount in legal_actions
            if action in [Action.BET, Action.RAISE]
        ]

        if not bet_actions:
            QMessageBox.warning(self, "Invalid Action", "Betting is not allowed right now.")
            return

        # Use the first bet/raise action's limits
        _, min_amount, max_amount = bet_actions[0]

        # Create bet dialog with style
        bet_dialog = QDialog(self)
        bet_dialog.setWindowTitle("Place Bet")
        bet_dialog.setStyleSheet("""
            QDialog {
                background-color: #2c5e17;
                color: white;
            }
            QLabel {
                color: white;
            }
            QSpinBox {
                background-color: rgba(255,255,255,0.2);
                color: white;
                border: 1px solid white;
                border-radius: 5px;
            }
        """)

        layout = QVBoxLayout()

        # Bet amount spinner
        bet_label = QLabel(f"Enter bet amount (${min_amount:.2f} - ${max_amount:.2f}):")
        layout.addWidget(bet_label)

        bet_spinner = QSpinBox()
        bet_spinner.setRange(int(min_amount), int(max_amount))
        bet_spinner.setValue(int(min_amount))
        layout.addWidget(bet_spinner)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(bet_dialog.accept)
        button_box.rejected.connect(bet_dialog.reject)
        layout.addWidget(button_box)

        bet_dialog.setLayout(layout)

        # Show dialog
        result = bet_dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            bet_amount = float(bet_spinner.value())
            self.player_action(Action.BET, bet_amount)

    def player_fold(self):
        """Handle player's fold action."""
        self.player_action