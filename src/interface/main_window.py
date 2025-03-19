import sys
from typing import Optional, Dict, Any

import torch
from PyQt6.QtWidgets import (
    QMainWindow, QApplication, QVBoxLayout, QHBoxLayout,
    QWidget, QPushButton, QLabel, QTabWidget, QTextEdit,
    QProgressBar, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

from src.algorithms.discounted_regret import DiscountedRegretMinimization
from src.engine.rules import PokerRules
from src.engine.evaluator import HandEvaluator
from config import DRM_PARAMS


class TrainingWorker(QThread):
    """Background thread for training the poker AI."""
    progress_update = pyqtSignal(dict)
    training_complete = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, drm: DiscountedRegretMinimization, num_iterations: int):
        super().__init__()
        self.drm = drm
        self.num_iterations = num_iterations

    def run(self):
        try:
            for iteration in range(1, self.num_iterations + 1):
                metrics = self.drm.iterate()
                metrics['iteration'] = iteration
                self.progress_update.emit(metrics)

                if iteration % 100 == 0:
                    exploitability = self.drm.compute_exploitability()
                    metrics['exploitability'] = exploitability
                    self.progress_update.emit(metrics)

            self.training_complete.emit(self.drm)
        except Exception as e:
            self.error_occurred.emit(str(e))


class PokerGTOMainWindow(QMainWindow):
    """Main window for the PokerGTO application."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PokerGTO - AI Poker Strategy Solver")
        self.resize(1200, 800)

        # Initialize core components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialize_drm_model()

        self.setup_ui()

    def initialize_drm_model(self):
        """Initialize the Discounted Regret Minimization model."""
        rules = PokerRules()
        evaluator = HandEvaluator()

        self.drm = DiscountedRegretMinimization(
            rules=rules,
            evaluator=evaluator,
            discount_factor=DRM_PARAMS['discount_factor'],
            device=self.device,
            batch_size=DRM_PARAMS['batch_size']
        )

    def setup_ui(self):
        """Set up the main user interface."""
        central_widget = QWidget()
        main_layout = QVBoxLayout()

        # Tabs for different functionalities
        tabs = QTabWidget()
        tabs.addTab(self.create_training_tab(), "Training")
        tabs.addTab(self.create_play_tab(), "Play")
        tabs.addTab(self.create_analysis_tab(), "Analysis")

        main_layout.addWidget(tabs)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def create_training_tab(self) -> QWidget:
        """Create the training tab with model training controls."""
        training_widget = QWidget()
        layout = QVBoxLayout()

        # Training controls
        training_controls = QHBoxLayout()
        iterations_btn = QPushButton("Train Model")
        iterations_btn.clicked.connect(self.start_training)
        training_controls.addWidget(iterations_btn)

        self.iterations_progress = QProgressBar()
        self.iterations_progress.setTextVisible(True)
        training_controls.addWidget(self.iterations_progress)

        layout.addLayout(training_controls)

        # Training log
        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        layout.addWidget(self.training_log)

        training_widget.setLayout(layout)
        return training_widget

    def create_play_tab(self) -> QWidget:
        """Create the play tab with play options."""
        play_widget = QWidget()
        # Placeholder for play functionality
        return play_widget

    def create_analysis_tab(self) -> QWidget:
        """Create the analysis tab with strategy analysis."""
        analysis_widget = QWidget()
        # Placeholder for strategy analysis
        return analysis_widget

    def start_training(self):
        """Start the model training process."""
        # Implement training with worker thread
        iterations = 10000  # Default training iterations

        self.training_worker = TrainingWorker(self.drm, iterations)
        self.training_worker.progress_update.connect(self.update_training_progress)
        self.training_worker.training_complete.connect(self.on_training_complete)
        self.training_worker.error_occurred.connect(self.on_training_error)

        self.training_worker.start()
        self.training_log.clear()

    def update_training_progress(self, metrics: Dict[str, Any]):
        """Update training progress UI."""
        iteration = metrics.get('iteration', 0)
        exploitability = metrics.get('exploitability', 0)

        self.iterations_progress.setValue(int((iteration / 10000) * 100))
        self.iterations_progress.setFormat(f"Iteration {iteration}: Exploitability {exploitability:.6f}")

        log_entry = f"Iteration {iteration}: " + \
                    f"Exploitability: {exploitability:.6f}\n"
        self.training_log.append(log_entry)

    def on_training_complete(self, trained_model):
        """Handle successful training completion."""
        QMessageBox.information(self, "Training Complete",
                                "Model training has been completed successfully!")

    def on_training_error(self, error_message: str):
        """Handle training errors."""
        QMessageBox.critical(self, "Training Error",
                             f"An error occurred during training: {error_message}")


def main():
    """Launch the PokerGTO application."""
    app = QApplication(sys.argv)
    main_window = PokerGTOMainWindow()
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()