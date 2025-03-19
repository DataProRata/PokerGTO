# PokerGTO

A Python project for developing Game Theory Optimal (GTO) Texas Hold'em poker strategies using advanced algorithms and machine learning techniques.

## Overview

PokerGTO implements Discounted Regret Minimization (DRM) to calculate optimal poker strategies and uses reinforcement learning to adapt these strategies through self-play and competition against the Slumbot API.

### Key Features

- **GTO Strategy Calculation**: Calculate equilibrium strategies using Discounted Regret Minimization
- **CUDA Acceleration**: Utilize GPU processing for faster computation
- **Adaptive Learning**: Train models against the Slumbot API to enhance performance
- **Interactive Play**: Test your skills against the trained models
- **Real-time Assistance**: Get GTO-based recommendations while playing

## Setup and Installation

### Prerequisites

- Python 3.8+ (via Anaconda)
- CUDA-compatible GPU
- Conda environment named "PokerGTO"

### Environment Setup

The project uses a Conda environment:

```bash
# Activate the existing PokerGTO environment
conda activate PokerGTO

# Install required dependencies
pip install -r requirements.txt
```

## Usage

The project consists of three main scripts:

### Calculate GTO Strategies

```bash
python calculate.py [--iterations 1000000] [--save_path models/drm_model]
```

This script calculates optimal poker strategies using Discounted Regret Minimization. The resulting model will be saved to the specified path.

### Train Against Slumbot

```bash
python train.py [--model_path models/drm_model] [--games 10000] [--save_path models/adaptive_model]
```

This script uses the initial DRM model to play against the Slumbot API, learning and adapting its strategy based on game outcomes.

### Interactive Play

```bash
python play.py [--model_path models/adaptive_model] [--mode interactive]
```

This script allows you to play against the trained models or get real-time GTO recommendations during play.

## Algorithm Details

### Discounted Regret Minimization

DRM is an extension of Counterfactual Regret Minimization (CFR) that applies a discount factor to accumulated regrets. This approach:

- Converges faster than vanilla CFR
- Adapts more quickly to strategic changes
- Performs better in large game trees like Texas Hold'em

### Adaptive Learning

The adaptive model extends the GTO baseline with reinforcement learning to counter exploitable tendencies in opponents.

## Project Structure

```
PokerGTO/
├── models/                      # Trained models storage
├── data/                        # Data storage
├── src/                         # Core source code
│   ├── engine/                  # Poker game engine
│   ├── algorithms/              # Algorithm implementations
│   ├── api/                     # API interactions
│   ├── utils/                   # Utility functions
│   └── interface/               # User interface
├── tests/                       # Unit and integration tests
├── calculate.py                 # Main GTO calculation script
├── train.py                     # Model training script
├── play.py                      # Interactive play script
└── config.py                    # Configuration settings
```

## References

- [Discounted CFR](https://arxiv.org/abs/1809.04040) - Brown & Sandholm
- [Deep Counterfactual Regret Minimization](https://arxiv.org/abs/1811.00164) - Brown et al.
- [Slumbot](http://www.slumbot.com/) - Poker AI competition bot

## License

This project is licensed for personal use only.