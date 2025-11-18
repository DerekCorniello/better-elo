import random
from typing import List
from deap import base, creator, tools
from .models import UserGameData


def predict_momentum_adjustment(weights, features: list) -> float:
    """Predict momentum adjustment for true Elo using evolved weights."""
    return float(sum(float(weights[i]) * float(features[i])
                     for i in range(len(features))))


def evaluate_individual(individual: list,
                        dataset: List[UserGameData]) -> tuple:
    """
    Evaluate fitness of an individual on the dataset (MSE for regression)
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
    # Add regularization penalty for large weights
    regularization = 0.01 * sum(abs(w) for w in individual)
    fitness = mse + regularization
    return (fitness,)


# using that deap package i found
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -500, 500)
toolbox.register("individual", tools.initRepeat,
                 creator.Individual, toolbox.attr_float, 6)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=5, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


def run_evolution(dataset: List[UserGameData], pop_size: int = 100,
                  ngen: int = 50, cxpb: float = 0.7,
                  mutpb: float = 0.2) -> tuple:
    """
    Run the evolutionary algorithm to find optimal feature weights
    """
    pop = toolbox.population(n=pop_size)
    # this is their elitism
    hof = tools.HallOfFame(1)
    fitnesses = list(map(lambda ind: toolbox.evaluate(ind, dataset), pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    hof.update(pop)

    # evolution loop
    for gen in range(ngen):
        # select offspring
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # apply crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                # TODO: since we removed the enforce_bounds func
                # idk if we stil need any of this or what...
                # Enforce bounds after crossover
                # child1[:] = enforce_bounds(child1)
                # child2[:] = enforce_bounds(child2)
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                # TODO: same as above
                # Enforce bounds after mutation
                # mutant[:] = enforce_bounds(mutant)
                del mutant.fitness.values

        # TODO: what does this do? idk what invalid means in this case
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
