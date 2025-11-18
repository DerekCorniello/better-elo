import random
from typing import List
from deap import base, creator, tools
from .models import UserGameData


def predict_momentum_adjustment(weights, features: list) -> float:
    """Predict momentum adjustment for true Elo using evolved weights."""
    return float(sum(float(weights[i]) * float(features[i]) for i in range(len(features))))


def evaluate_individual(individual: list, dataset: List[UserGameData]) -> tuple:
    """Evaluate fitness of an individual on the dataset (MSE for regression)."""
    total_error = 0.0
    total = len(dataset)
    sample_count = 0
    for game in dataset:
        adjustment = predict_momentum_adjustment(
            individual, game.to_feature_vector())
        predicted_adjusted_elo = game.pre_game_elo + adjustment
        actual_elo = game.post_game_elo
        error = (predicted_adjusted_elo - actual_elo) ** 2
        total_error += error
        # Store adjustment for interpretation
        game.momentum_adjustment = adjustment
        # Debug: print first 3 samples
        if sample_count < 3:
            print(f"Eval sample {sample_count}: pre={game.pre_game_elo:.1f}, adj={adjustment:.3f}, pred={predicted_adjusted_elo:.1f}, actual={actual_elo:.1f}")
            sample_count += 1
    mse = total_error / total if total > 0 else 0.0
    # Add regularization penalty for large weights
    regularization = 0.01 * sum(abs(w) for w in individual)
    fitness = mse + regularization
    return (fitness,)


# Define the optimization problem
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Create toolbox
toolbox = base.Toolbox()

# Attribute generator: real-valued weights with expanded range for unconstrained optimization
toolbox.register("attr_float", random.uniform, -500, 500)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 6)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def enforce_bounds(individual, lower=None, upper=None):
    """No bounds enforcement - allow true optimal weights."""
    return individual

# Genetic operators
toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Blend crossover with moderate exploration
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=5, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


def run_evolution(dataset: List[UserGameData], pop_size: int = 100, ngen: int = 50, cxpb: float = 0.7, mutpb: float = 0.2) -> tuple:
    """Run the evolutionary algorithm to find optimal feature weights."""
    # Create population
    pop = toolbox.population(n=pop_size)

    # Hall of fame for elitism
    hof = tools.HallOfFame(1)

    # Evaluate initial population
    fitnesses = list(map(lambda ind: toolbox.evaluate(ind, dataset), pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    hof.update(pop)

    # Evolution loop
    for gen in range(ngen):
        # Select offspring
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                # Enforce bounds after crossover
                child1[:] = enforce_bounds(child1)
                child2[:] = enforce_bounds(child2)
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                # Enforce bounds after mutation
                mutant[:] = enforce_bounds(mutant)
                del mutant.fitness.values

        # Evaluate offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(lambda ind: toolbox.evaluate(ind, dataset), invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update hall of fame
        hof.update(offspring)

        # Debug: print progress every 5 generations
        if gen % 5 == 0:
            best_fitness = hof[0].fitness.values[0]
            print(f"Gen {gen}: Best fitness (MSE) = {best_fitness:.3f}")

        # Replace population
        pop[:] = offspring

    # Return best individual
    best = hof[0]
    return best
