# 🃏 PokerGTO

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue.svg" alt="Python 3.12"/>
  <img src="https://img.shields.io/badge/PyTorch-2.2.0-orange.svg" alt="PyTorch 2.2.0"/>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"/>
  <img src="https://img.shields.io/badge/CUDA-Enabled-brightgreen.svg" alt="CUDA Enabled"/>
</p>

<p align="center">
  <em>Advanced Game Theory Optimal (GTO) solver for Texas Hold'em poker using Discounted Regret Minimization with GPU acceleration</em>
</p>

---

## 📋 Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Installation Guide](#-installation-guide)
- [Usage Guide](#-usage-guide)
- [Algorithm Details](#-algorithm-details)
- [Development Roadmap](#-development-roadmap)
- [References](#-references)

---

## 🔍 Overview

PokerGTO is a Python project for developing Game Theory Optimal (GTO) poker strategies using advanced algorithms and machine learning techniques. The system calculates equilibrium solutions using Discounted Regret Minimization (DRM) and can leverage CUDA-enabled GPUs for accelerated computation.

The project includes functionality to calculate baseline GTO strategies, train adaptive models against the Slumbot API, and provide real-time strategy recommendations during gameplay.

<p align="center">
  <img src="docs/images/strategy_sample.png" alt="Strategy Visualization" width="80%"/>
  <br>
  <em>Sample strategy visualization (image will be generated during training)</em>
</p>

---

## 🌟 Key Features

- **Advanced GTO Computation**: Calculate Nash equilibrium strategies using Discounted Regret Minimization
- **GPU Acceleration**: Utilize CUDA for dramatically faster computation
- **Adaptive Learning**: Train models against the Slumbot API to enhance performance
- **Interactive Play**: Test your skills against the trained models
- **Real-time Assistance**: Get GTO-based recommendations while playing

---

## 📂 Project Structure

```
PokerGTO/
├── models/                      # Trained models storage
│   ├── drm_model/               # Discounted Regret Minimization model
│   └── adaptive_model/          # Model learning from Slumbot interactions
│
├── data/                        # Data storage
│   ├── game_logs/               # Logs of games played
│   └── training_history/        # Training metrics and visualization
│
├── src/                         # Core source code
│   ├── engine/                  # Poker game engine
│   ├── algorithms/              # Algorithm implementations
│   ├── api/                     # API interactions
│   ├── utils/                   # Utility functions
│   └── interface/               # User interface
│
├── tests/                       # Unit and integration tests
├── calculate.py                 # GTO calculation script
├── train.py                     # Model training script
├── play.py                      # Interactive play script
├── config.py                    # Configuration settings
├── setup_poker_env.ps1          # Environment setup script
└── requirements.txt             # Project dependencies
```

---

## 💻 Installation Guide

### Prerequisites

- Python 3.8+ (via Anaconda)
- CUDA-compatible GPU (recommended)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/PokerGTO.git
cd PokerGTO
```

### Step 2: Create and Configure the Conda Environment

We provide a PowerShell script that sets up the environment with all required dependencies. The script creates a new conda environment named "PokerGTO" with Python 3.12 and installs all necessary packages.

```powershell
# Run the setup script (in PowerShell)
.\setup_poker_env.ps1
```

If you encounter execution policy restrictions, you can run:

```powershell
PowerShell -ExecutionPolicy Bypass -File .\setup_poker_env.ps1
```

#### Manual Environment Setup

If you prefer to set up the environment manually or are using a non-Windows system:

```bash
# Create conda environment with Python 3.12
conda create -n PokerGTO python=3.12 -y
conda activate PokerGTO

# Install core packages
conda install numpy pandas matplotlib tensorboard requests jupyter scikit-learn seaborn -y

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install poker-specific packages
pip install treys  # Python poker hand evaluation library

# Install other requirements
pip install tqdm pytest python-dotenv
```

### Step 3: Verify Installation

To verify your installation:

```bash
conda activate PokerGTO
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('CUDA Version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
```

This should output your CUDA availability and version information.

---

## 🎮 Usage Guide

### Calculate GTO Strategies

The `calculate.py` script runs the Discounted Regret Minimization algorithm to compute Game Theory Optimal (GTO) strategies:

```bash
# Activate the PokerGTO environment
conda activate PokerGTO

# Run with default settings
python calculate.py

# Run with custom settings
python calculate.py --iterations 1000000 --discount 0.95 --save_path models/my_drm_model
```

Key parameters:
- `--iterations`: Number of iterations to run (default: 1,000,000)
- `--discount`: Discount factor for regrets (default: 0.95)
- `--save_path`: Directory to save the model (default: models/drm_model)
- `--no_cuda`: Disable CUDA acceleration even if available

### Train Against Slumbot

The `train.py` script uses the initial DRM model to play against Slumbot and refine its strategy:

```bash
# Train with default settings
python train.py

# Train with custom settings
python train.py --model_path models/drm_model --games 10000 --save_path models/my_adaptive_model
```

Key parameters:
- `--model_path`: Path to the pre-trained DRM model (default: models/drm_model) 
- `--games`: Number of games to play (default: 10,000)
- `--save_path`: Directory to save the trained model (default: models/adaptive_model)

### Interactive Play

The `play.py` script allows you to play poker with real-time GTO strategy recommendations:

```bash
# Play with default settings
python play.py

# Play with custom settings
python play.py --model_path models/adaptive_model --mode interactive
```

Key parameters:
- `--model_path`: Path to the trained model (default: models/adaptive_model)
- `--mode`: Play mode - interactive, auto, or tooltips (default: interactive)

---

## 📊 Algorithm Details

### Discounted Regret Minimization

Discounted Regret Minimization (DRM) is an extension of Counterfactual Regret Minimization (CFR) that applies a discount factor to accumulated regrets. This approach has several advantages:

- **Faster Convergence**: Converges to Nash equilibrium more quickly than vanilla CFR
- **Better Adaptation**: Responds more effectively to strategic changes
- **Improved Performance**: Handles large game trees more efficiently

DRM applies the following update rule for cumulative regrets:

```
R_t+1(I,a) = γ·R_t(I,a) + r_t(I,a)
```

Where:
- `R_t(I,a)` is the cumulative regret for information state I and action a at time t
- `γ` is the discount factor (typically 0.5-0.95)
- `r_t(I,a)` is the counterfactual regret at time t

### Adaptive Learning

The adaptive model extends the GTO baseline with reinforcement learning to counter exploitable tendencies in opponents:

1. **Baseline Strategy**: Start with the GTO strategy calculated by DRM
2. **Opponent Modeling**: Track opponent tendencies during gameplay
3. **Strategy Adaptation**: Adjust strategy to exploit identified weaknesses
4. **Continual Learning**: Update the model as more games are played

---

## 🛣️ Development Roadmap

- [ ] Enhance hand evaluator performance
- [ ] Add neural network models for improved learning
- [ ] Implement more sophisticated opponent modeling
- [ ] Develop a graphical user interface
- [ ] Expand to multi-way pots and tournaments
- [ ] Add support for other poker variants

---

## 📚 References

- [Discounted CFR](https://arxiv.org/abs/1809.04040) - Brown & Sandholm (2018)
- [Deep Counterfactual Regret Minimization](https://arxiv.org/abs/1811.00164) - Brown et al. (2018)
- [Slumbot](http://www.slumbot.com/) - Poker AI competition bot
- [Libratus](https://www.cs.cmu.edu/~noamb/papers/17-arXiv-Libratus.pdf) - First AI to defeat top poker professionals

---

<p align="center">
  <em>Created with ❤️ for poker and AI</em>
</p>