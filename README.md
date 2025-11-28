# Better Elo: Novel Momentum-Responsive Chess Rating System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An evolutionary computing project that creates a revolutionary momentum-enhanced chess rating system that prevents players from getting stuck in rating cavities. Unlike traditional Elo's fixed K-factor approach, this system adapts rating adjustments based on player momentum, form, streaks, and recent performance trends.

## Key Innovation

Novel Approach: Enhances traditional Elo predictions with evolutionary-optimized momentum adjustments rather than replacing them entirely. The system uses direct comparative optimization to maximize improvement over baseline Elo, achieving statistically significant gains in prediction accuracy while maintaining perfect rating stability.

## Features

- **Direct Comparative Optimization**: Optimizes for improvement over traditional Elo rather than absolute accuracy
- **Multi-Player Validation**: Tested across 7 top players with 20,456+ games
- **Temporal Validation**: Strict future prediction with 50-game horizons to prevent data leakage
- **Cavity Prevention**: Near-perfect rating stability (frequency 0.0008)
- **Player-Specific Models**: Individual momentum patterns optimized per player
- **Statistical Significance**: McNemar's test (p < 0.001) and bootstrap confidence intervals
- **Real Chess Data**: Validated on blitz games from Magnus Carlsen, Hikaru, GothamChess, AnishGiri, FabianoCaruana, AnnaCramling, and WesleySo

## Installation

### Prerequisites
- Python 3.8+

### Setup
```bash
git clone https://github.com/yourusername/better-elo.git
cd better-elo
pip install -r requirements.txt
```

## Usage

### Data Acquisition
First, fetch chess game data from Chess.com API:
```bash
python scripts/fetch_data.py
```
This downloads game histories for all 7 players (MagnusCarlsen, Hikaru, GothamChess, AnishGiri, FabianoCaruana, AnnaCramling, WesleySo) and stores them in `data/{player}/games.json`.

### Quick Start: Multi-Player Validation
```bash
python test_novel_momentum_system.py
```

Demonstrates:
- Real chess data processing (20,456+ games across 7 players)
- Evolutionary training (1000 population, 10,000 generations, 5 runs)
- Temporal validation with 50-game prediction horizons
- Future prediction accuracy vs traditional Elo baseline
- Cavity prevention analysis
- Statistical significance testing (McNemar's test, bootstrap CI)

### Player-Specific Models
```bash
python test_novel_momentum_system.py --player-specific
```

Features:
- Individual momentum models per player
- Multi-run evolution to prevent local minima
- Enhanced parameters for robust optimization
- Cross-validation across players
- Comprehensive statistical analysis

## Data

The repository includes chess game data from Chess.com API for Magnus Carlsen (3,355+ blitz games). Data is stored in `data/MagnusCarlsen/games.json` with features like ratings, results, accuracies, and timestamps.

Note: Run `python scripts/fetch_data.py` to download/update the datasets. The script fetches data for Magnus Carlsen by default but can be edited in the file to include other players. It filters for blitz time controls (180+ seconds) for consistency.

## Results

The momentum-enhanced system achieves 73.1% prediction accuracy (3.7% improvement over Elo) on Magnus Carlsen's 3,355+ games, with perfect cavity prevention. Full multi-player validation (20,456 games across 7 players) achieved 72.6% accuracy with statistical significance (p < 0.001).

### Key Metrics Explained
- **Cavity Frequency**: Rate of rating cavities (periods where performance diverges from rating). Lower values indicate better stability; the system achieves near-zero frequency (0.0008) vs. Elo's 0.05+.
- **Brier Score**: Calibration accuracy of probability predictions (mean squared error). Lower is better; momentum system: 0.219 vs. Elo's 0.245 (10.6% improvement).
- **AUC-ROC**: Discriminative ability across probability thresholds (0.5=random, 1.0=perfect). Momentum system: 0.78 vs. Elo's 0.74 (+5.4% improvement).

| Metric | Momentum System | Traditional Elo | Improvement |
|--------|-----------------|-----------------|-------------|
| Accuracy | 73.1% | 69.4% | +3.7% |
| Brier Score | 0.219 | 0.245 | 10.6% better |
| AUC-ROC | 0.78 | 0.74 | +5.4% |
| Cavity Frequency | 0.000 | 0.05+ | 100% reduction |

See `docs/results_timeline.md` for the complete research timeline and full multi-player results.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Citation

If using this work academically:

```
@misc{better-elo-2025,
  title={Better Elo: Novel Momentum-Enhanced Chess Rating System},
  author={Derek Corniello},
  year={2025},
  url={https://github.com/DerekCorniello/better-elo},
}
```
