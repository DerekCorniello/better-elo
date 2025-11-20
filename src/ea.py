import random
from typing import List
from deap import base, creator, tools
import sys
import os
# Add current directory to path for absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from models import UserGameData


def predict_momentum_adjustment(weights, features: list) -> float:
    """Predict momentum adjustment for true Elo using evolved weights."""
    return float(sum(float(weights[i]) * float(features[i])
                     for i in range(len(features))))


def evaluate_individual(individual: list,
                         dataset: List[UserGameData]) -> tuple:
    """
    Evaluate fitness of an individual on the dataset (MSE for Elo prediction)
    """
    total_error = 0.0
    total = len(dataset)
    for game in dataset:
        adjustment = predict_momentum_adjustment(
            individual, game.to_feature_vector())
        predicted_adjusted_elo = game.pre_game_elo + adjustment
        actual_elo = game.post_game_elo
        error = (predicted_adjusted_elo - actual_elo) ** 2
        total_error += error
        # store adjustment for interpretation
        game.momentum_adjustment = adjustment
    mse = total_error / total if total > 0 else 0.0
    # Add L2 regularization penalty for large weights
    regularization = 0.001 * sum(w**2 for w in individual)
    fitness = mse + regularization
    return (fitness,)


# using that deap package i found
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -5, 5)
toolbox.register("individual", tools.initRepeat,
                 creator.Individual, toolbox.attr_float, 6)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=3, indpb=0.3)  # Higher mutation rate with elitism
toolbox.register("select", tools.selTournament, tournsize=7)  # Larger tournament


def run_evolution(dataset: List[UserGameData], pop_size: int = 100,
                  ngen: int = 50, cxpb: float = 0.8,
                  mutpb: float = 0.3) -> tuple:
    """
    Run the evolutionary algorithm to find optimal feature weights
    """
    pop = toolbox.population(n=pop_size)
    # Elitism: retain top 10% of population
    hof = tools.HallOfFame(int(pop_size * 0.1))
    fitnesses = list(map(lambda ind: toolbox.evaluate(ind, dataset), pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    hof.update(pop)

    # evolution loop
    for gen in range(ngen):
        # Progress reporting
        if gen % 20 == 0 or gen == ngen - 1:
            best_fitness = hof[0].fitness.values[0] if hof else float('inf')
            print(f"Generation {gen+1}/{ngen}: Best fitness = {best_fitness:.4f}")

        # select offspring (elitism: keep best individuals)
        offspring = toolbox.select(pop, len(pop) - len(hof))
        offspring = list(map(toolbox.clone, offspring))
        offspring.extend(toolbox.clone(ind) for ind in hof)  # add elite individuals

        # apply crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Adaptive mutation: decrease sigma over generations for exploration to exploitation
        sigma = 3 * (0.1 + 0.9 * (ngen - gen) / ngen)
        toolbox.unregister("mutate")
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=sigma, indpb=0.3)

        # Apply mutation
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate invalid individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(
            map(lambda ind: toolbox.evaluate(ind, dataset), invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        hof.update(offspring)

        # Replace population
        pop[:] = offspring

    # Return best individual
    best = hof[0]
    return best
