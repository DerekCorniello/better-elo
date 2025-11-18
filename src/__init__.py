from .models import PlayerFeatures, MatchData, UserGameData
from .data_generator import RealDataGenerator
from .ea import predict_momentum_adjustment, evaluate_individual, run_evolution
from .evaluation import train_test_split, evaluate_baseline, run_evaluation, statistical_analysis

__version__ = "0.1.0"