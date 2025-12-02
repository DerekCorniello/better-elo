import os
import random
import sys
from typing import List

from deap import base, creator, tools

from .config import config
from .logging_config import get_logger
from .models import UserGameData

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

logger = get_logger(__name__)


# clean up existing DEAP creator classes to avoid conflicts
try:
    if hasattr(creator, "FitnessMin"):
        del creator.FitnessMin
    if hasattr(creator, "Individual"):
        del creator.Individual
except Exception:
    pass


def predict_momentum_adjustment(weights, features: list) -> float:
    """Predict momentum adjustment for true Elo using evolved weights.

    Args:
        weights: List of evolved weight values.
        features: List of feature values for momentum calculation.

    Returns:
        float: Predicted momentum adjustment.
    """
    return float(
        sum(
            float(weights[i]) * float(features[i])
            for i in range(min(len(weights), len(features)))
        )
    )


def calculate_momentum_prediction(game, momentum_weights):
    """Calculate momentum-enhanced prediction for a single game.

    Args:
        game: UserGameData object containing game information.
        momentum_weights: List of weights for momentum calculation.

    Returns:
        float: Enhanced prediction probability.
    """
    # traditional Elo prediction
    elo_expected = 1 / (
        1 + 10 ** ((game.opponent_elo - game.pre_game_elo) / 400)
    )

    # momentum adjustment
    features = game.to_feature_vector()
    momentum_adjustment = sum(w * f for w, f in zip(momentum_weights, features))
    momentum_adjustment = max(-0.2, min(0.2, momentum_adjustment))

    # enhanced prediction
    enhanced_prob = elo_expected + momentum_adjustment
    enhanced_prob = max(0.01, min(0.99, enhanced_prob))

    return enhanced_prob

def evaluate_individual(
    individual: list, dataset: List[UserGameData]
) -> tuple:
    """Evaluate fitness by direct comparison: Momentum system vs traditional Elo.

    Optimizes for better matchmaking through improved prediction accuracy.

    Args:
        individual: List of weight values for momentum calculation.
        dataset: List of UserGameData objects for evaluation.

    Returns:
        tuple: Fitness value (negative improvement + regularization).
    """
    momentum_correct = 0
    elo_correct = 0
    total_games = len(dataset)

    for game in dataset:
        # calculate momentum prediction
        momentum_prob = calculate_momentum_prediction(game, individual)
        momentum_win = 1 if momentum_prob > 0.5 else 0

        # calculate Elo prediction
        elo_prob = 1 / (
            1 + 10 ** ((game.opponent_elo - game.pre_game_elo) / 400)
        )
        elo_win = 1 if elo_prob > 0.5 else 0

        actual_win = 1 if game.actual_result > 0.5 else 0
        # count correct predictions
        if momentum_win == actual_win:
            momentum_correct += 1
        if elo_win == actual_win:
            elo_correct += 1

    # calculate accuracies
    momentum_accuracy = momentum_correct / total_games
    elo_accuracy = elo_correct / total_games

    # add much stronger L2 regularization penalty for large weights
    # stronger to prevent weight explosion
    regularization = 0.001 * sum(w**2 for w in individual)

    # fitness is a measure of direct improvement over Elo
    # (negative for DEAP minimization)
    improvement = momentum_accuracy - elo_accuracy
    return (-improvement + regularization,)


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -50, 50)
toolbox.register(
    "individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 6
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_individual)


def differential_evolution_mutation(pop, F, CR):
    """DE/rand/1/bin mutation strategy.

    Args:
        pop: Current population of individuals.
        F: Differential weight parameter.
        CR: Crossover rate parameter.

    Returns:
        list: New population after mutation and crossover.
    """
    new_pop = []
    n = len(pop)
    dim = len(pop[0])

    for i in range(n):
        # select three random individuals (r1, r2, r3) different from current
        candidates = [j for j in range(n) if j != i]
        r1, r2, r3 = random.sample(candidates, 3)

        # create mutant vector withv = r1 + F * (r2 - r3)
        mutant = []
        for d in range(dim):
            mutant_val = pop[r1][d] + F * (pop[r2][d] - pop[r3][d])
            mutant.append(mutant_val)

        # Crossover: binomial crossover
        trial = []
        for d in range(dim):
            if random.random() < CR or d == random.randint(0, dim - 1):
                trial.append(mutant[d])
            else:
                trial.append(pop[i][d])

        # apply weight bounds to prevent explosion
        trial = [max(-100, min(100, val)) for val in trial]
        new_pop.append(trial)

    return new_pop


def run_evolution(
    dataset: List[UserGameData],
    pop_size: int | None = None,
    ngen: int | None = None,
    cxpb: float = 0.8,
    mutpb: float = 0.3,
) -> tuple:
    """Run Differential Evolution algorithm to find optimal feature weights.

    DE/rand/1/bin strategy with self-adaptive parameters.

    Args:
        dataset: List of UserGameData objects for training.
        pop_size: Population size (uses config if None).
        ngen: Number of generations (uses config if None).
        cxpb: Crossover probability.
        mutpb: Mutation probability.

    Returns:
        tuple: Best individual found during evolution.
    """
    # Use config values if not provided
    if pop_size is None:
        pop_size = config.evolution["population_size"]
    if ngen is None:
        ngen = config.evolution["generations"]

    pop = toolbox.population(n=pop_size)
    fitnesses = []
    for ind in pop:
        fit = toolbox.evaluate(ind, dataset)
        ind.fitness.values = fit
        fitnesses.append(fit[0])

    # this is what they call elitism
    hof = tools.HallOfFame(int(pop_size * 0.05))
    hof.update(pop)

    # DE parameters with self-adaptation
    F = config.evolution["f_value"]
    CR = config.evolution["cr_value"]
    F_min, F_max = 0.3, 0.9
    CR_min, CR_max = 0.5, 0.9

    # evolution loop with early convergence detection
    best_fitness_history = []
    no_improvement_count = 0

    for gen in range(ngen):
        if gen % 50 == 0 or gen == ngen - 1:
            best_fitness = hof[0].fitness.values[0] if hof else float("inf")
            logger.info(
                "Generation %d/%d: Best fitness = %.4f (F=%.3f, CR=%.3f)",
                gen + 1,
                ngen,
                best_fitness,
                F,
                CR,
            )

            # Track fitness for convergence detection
            best_fitness_history.append(best_fitness)

            # Check for convergence
            if len(best_fitness_history) >= 2:
                improvement = (
                    best_fitness_history[-2] - best_fitness_history[-1]
                )
                if improvement < 0.0001:
                    no_improvement_count += 1
                    if no_improvement_count >= 2:
                        logger.info(
                            "DE converged after %d generations "
                            "(improvement < 0.0001 for 100 generations)",
                            gen + 1,
                        )
                        break
                else:
                    no_improvement_count = 0

        # self adaptive DE parameters based on generation progress
        progress = gen / ngen
        F = F_min + (F_max - F_min) * (1 - progress)  # start high, end low
        CR = CR_min + (CR_max - CR_min) * progress  # start low, end high

        # Generate trial population using DE/rand/1/bin
        trial_pop_raw = differential_evolution_mutation(pop, F, CR)

        # Evaluate trial population and create proper individuals
        trial_pop = []
        trial_fitnesses = []
        for trial_ind in trial_pop_raw:
            new_ind = creator.Individual(trial_ind)
            fit = toolbox.evaluate(new_ind, dataset)
            new_ind.fitness.values = fit
            trial_pop.append(new_ind)
            trial_fitnesses.append(fit[0])

        # selection is to replace current with trial if better
        new_pop = []
        new_fitnesses = []
        for i in range(len(pop)):
            # better fitness, lower is better
            if trial_fitnesses[i] < fitnesses[i]:
                new_pop.append(trial_pop[i])
                new_fitnesses.append(trial_fitnesses[i])
            else:
                new_pop.append(pop[i])
                new_fitnesses.append(fitnesses[i])

        # this is how to update elitism, super nice
        hof.update(new_pop)

        # replace worst individuals with elites if needed
        if len(hof) > 0:
            # sort population by fitness
            sorted_indices = sorted(
                range(len(new_pop)), key=lambda i: new_fitnesses[i]
            )

            # replace worst individuals with elites
            for i, elite in enumerate(hof):
                if i < len(sorted_indices):
                    worst_idx = sorted_indices[i]
                    new_pop[worst_idx] = creator.Individual(list(elite))
                    new_pop[worst_idx].fitness.values = elite.fitness.values
                    new_fitnesses[worst_idx] = elite.fitness.values[0]

        pop = new_pop
        fitnesses = new_fitnesses

    return hof[0]
