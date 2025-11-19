# Better Elo: Novel Momentum-Responsive Chess Rating System

An evolutionary computing project that creates a **revolutionary momentum-based rating system** that prevents players from getting stuck in rating cavities. Unlike chess.com's fixed K-factor approach, this system adapts rating adjustments based on player form, streaks, and recent performance trends.

## 🎯 Key Innovation

**Novel Approach**: Creates independent ratings that compete with traditional Elo by using momentum as leading indicators of chess performance.

## Usage

### 🎯 Novel Momentum System - Single Player (Magnus Carlsen)
```bash
python test_novel_momentum_system.py
```

**Demonstrates:**
- ✅ **Real chess data processing** (3,355+ games from Magnus Carlsen)
- ✅ **Intensive evolutionary training** (150 population, 200 generations)
- ✅ **Aggressive evolution parameters** (higher mutation/crossover rates)
- ✅ **Temporal validation** with 50-game prediction horizons
- ✅ **Future prediction accuracy** measurement vs 50% baseline
- ✅ **Cavity prevention analysis** on real rating trajectories
- ✅ **Statistical comparison** vs traditional Elo system

### 🎯 Novel Momentum System - Multi-Player Cross-Validation
```bash
python test_novel_momentum_system.py --multi-player
```

**Demonstrates:**
- ✅ **Cross-player validation** across Anna Cramling, hikaru, Fabiano Caruana, Magnus Carlsen
- ✅ **Intensive multi-player training** (100 population, 100 generations per fold)
- ✅ **Universal momentum patterns** that work across different players
- ✅ **Generalized cavity prevention** for all skill levels
- ✅ **Statistical robustness** across diverse player profiles
- ✅ **Proves momentum system superiority** universally, not just for champions
