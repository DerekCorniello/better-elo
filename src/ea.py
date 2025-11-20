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
    Evaluate fitness of an individual using momentum-enhanced Elo predictions
    """
    import math
    total_error = 0.0
    total = len(dataset)
    for game in dataset:
        # Calculate traditional Elo prediction
        elo_expected = 1 / (1 + 10 ** ((game.opponent_elo - game.pre_game_elo) / 400))
        
        # Calculate momentum adjustment (small adjustment to Elo)
        features = game.to_feature_vector()
        momentum_adjustment = sum(w * f for w, f in zip(individual, features))
        
        # Limit momentum adjustment to reasonable range (-0.2 to +0.2)
        momentum_adjustment = max(-0.2, min(0.2, momentum_adjustment))
        
        # Enhanced prediction: Elo + momentum adjustment
        enhanced_prob = elo_expected + momentum_adjustment
        
        # Ensure probability stays in valid range [0, 1]
        enhanced_prob = max(0.01, min(0.99, enhanced_prob))
        
        actual_result = game.actual_result
        error = (enhanced_prob - actual_result) ** 2
        total_error += error
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

    # evolution loop with early convergence detection
    best_fitness_history = []
    no_improvement_count = 0
    
    for gen in range(ngen):
        # Progress reporting every 50 generations
        if gen % 50 == 0 or gen == ngen - 1:
            best_fitness = hof[0].fitness.values[0] if hof else float('inf')
            print(f"Generation {gen+1}/{ngen}: Best fitness = {best_fitness:.4f}")
            
            # Track fitness for convergence detection
            best_fitness_history.append(best_fitness)
            
            # Check for convergence (no improvement < 0.0001 in last 100 generations)
            if len(best_fitness_history) >= 2:
                improvement = best_fitness_history[-2] - best_fitness_history[-1]
                if improvement < 0.0001:
                    no_improvement_count += 1
                    if no_improvement_count >= 2:  # 100 generations with no improvement (50*2)
                        print(f"Converged after {gen+1} generations (improvement < 0.0001 for 100 generations)")
                        break
                else:
                    no_improvement_count = 0

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
