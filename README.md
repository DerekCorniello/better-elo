# Better Elo: Novel Momentum-Responsive Chess Rating System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An evolutionary computing project that creates a revolutionary momentum-based rating system that prevents players from getting stuck in rating cavities. Unlike chess.com's fixed K-factor approach, this system adapts rating adjustments based on player form, streaks, and recent performance trends.

## Key Innovation

Novel Approach: Creates independent ratings that compete with traditional Elo by using momentum as leading indicators of chess performance. The system enhances Elo predictions with evolutionary-optimized momentum adjustments, achieving measurable improvements in accuracy and calibration while preventing rating stagnation.

## Features

- Evolutionary Optimization: Uses DEAP genetic algorithms to discover optimal momentum weights
- Temporal Validation: Strict future prediction with 50-game horizons to prevent data leakage
- Cavity Prevention: Quantifies and minimizes rating stagnation periods
- Player-Specific Models: Individual momentum patterns optimized per player
- Statistical Significance: Bootstrap confidence intervals and McNemar's test
- Real Chess Data: Validated on 3,355+ blitz games from Magnus Carlsen

## Installation

### Prerequisites
- Python 3.8+
- Dependencies: DEAP, NumPy, SciPy

### Setup
```bash
git clone https://github.com/yourusername/better-elo.git
cd better-elo
pip install deap numpy scipy
```

## Usage

### Quick Start: Magnus Carlsen Demo
```bash
python test_novel_momentum_system.py
```

Demonstrates:
- Real chess data processing (3,355+ games)
- Evolutionary training (200 population, 200 generations)
- Temporal validation with 50-game prediction horizons
- Future prediction accuracy vs 50% baseline
- Cavity prevention analysis
- Statistical comparison vs traditional Elo

### Player-Specific Models
```bash
python test_novel_momentum_system.py --player-specific
```

Features:
- Individual momentum models per player
- Multi-run evolution to prevent local minima
- Enhanced parameters (1000 population, 10,000 generations)
- Cross-validation across players

## Data

The repository includes chess game data from Chess.com API for Magnus Carlsen (3,355 blitz games). Data is stored in `data/MagnusCarlsen/games.json` with features like ratings, results, and timestamps.

Note: Other player data is excluded via `.gitignore` for repository size. Contact the authors for access to full datasets.

## Results

The momentum system achieves 67.5% prediction accuracy (2.1% improvement over Elo) with near-perfect cavity prevention (frequency 0.001). Key metrics:

| Metric | Momentum System | Traditional Elo | Improvement |
|--------|-----------------|-----------------|-------------|
| Accuracy | 67.5% | 65.3% | +2.1% |
| Brier Score | 0.149 | 0.174 | 14.7% better |
| Cavity Frequency | 0.001 | 0.05+ | 98% reduction |

See `docs/` for detailed results and methodology.

## Documentation

- [Current Implementation Summary](docs/current_implementation_summary.md)
- [Research Summary](docs/research_summary.md)
- [Final Results](docs/FINAL_MOMENTUM_SYSTEM_RESULTS.md)
- [Player-Specific Findings](docs/player_specific_findings_v4.md)

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Citation

If using this work academically:

```
@misc{better-elo-2024,
  title={Better Elo: Novel Momentum-Responsive Chess Rating System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/better-elo}
}
```
