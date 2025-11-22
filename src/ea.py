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

# Clean up existing DEAP creator classes to avoid conflicts
try:
    if hasattr(creator, "FitnessMin"):
        del creator.FitnessMin
    if hasattr(creator, "Individual"):
        del creator.Individual
except:
    pass

def predict_momentum_adjustment(weights, features: list) -> float:
    """Predict momentum adjustment for true Elo using evolved weights."""
    return float(sum(float(weights[i]) * float(features[i])
                     for i in range(min(len(weights), len(features)))))


def calculate_momentum_prediction(game, momentum_weights):
    """
    Calculate momentum-enhanced prediction for a single game
    """
    # Traditional Elo prediction
    elo_expected = 1 / (1 + 10 ** ((game.opponent_elo - game.pre_game_elo) / 400))
    
    # Momentum adjustment
    features = game.to_feature_vector()
    momentum_adjustment = sum(w * f for w, f in zip(momentum_weights, features))
    momentum_adjustment = max(-0.2, min(0.2, momentum_adjustment))
    
    # Enhanced prediction
    enhanced_prob = elo_expected + momentum_adjustment
    enhanced_prob = max(0.01, min(0.99, enhanced_prob))
    
    return enhanced_prob


def evaluate_individual(individual: list,
                         dataset: List[UserGameData]) -> tuple:
    """
    Evaluate fitness by direct comparison: Momentum system vs traditional Elo
    Optimizes for better matchmaking through improved prediction accuracy
    """
    momentum_correct = 0
    elo_correct = 0
    total_games = len(dataset)
    
    for game in dataset:
        # Calculate momentum prediction
        momentum_prob = calculate_momentum_prediction(game, individual)
        momentum_win = 1 if momentum_prob > 0.5 else 0
        
        # Calculate Elo prediction
        elo_prob = 1 / (1 + 10 ** ((game.opponent_elo - game.pre_game_elo) / 400))
        elo_win = 1 if elo_prob > 0.5 else 0
        
        # Determine actual result
        actual_win = 1 if game.actual_result > 0.5 else 0
        
        # Count correct predictions
        if momentum_win == actual_win:
            momentum_correct += 1
        if elo_win == actual_win:
            elo_correct += 1
    
    # Calculate accuracies
    momentum_accuracy = momentum_correct / total_games
    elo_accuracy = elo_correct / total_games
    
    # Add much stronger L2 regularization penalty for large weights
    regularization = 0.001 * sum(w**2 for w in individual)  # 100x stronger to prevent weight explosion
    
    # Fitness: Direct improvement over Elo (negative for DEAP minimization)
    improvement = momentum_accuracy - elo_accuracy
    return (-improvement + regularization,)


# Create DEAP classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -50, 50)  # Larger initial range but will be bounded
toolbox.register("individual", tools.initRepeat,
                 creator.Individual, toolbox.attr_float, 6)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_individual)


def differential_evolution_mutation(pop, F, CR):
    """
    DE/rand/1/bin mutation strategy
    """
    new_pop = []
    n = len(pop)
    dim = len(pop[0])
    
    for i in range(n):
        # Select three random individuals (r1, r2, r3) different from current
        candidates = [j for j in range(n) if j != i]
        r1, r2, r3 = random.sample(candidates, 3)
        
        # Create mutant vector: v = r1 + F * (r2 - r3)
        mutant = []
        for d in range(dim):
            mutant_val = pop[r1][d] + F * (pop[r2][d] - pop[r3][d])
            mutant.append(mutant_val)
        
        # Crossover: binomial crossover
        trial = []
        for d in range(dim):
            if random.random() < CR or d == random.randint(0, dim-1):
                trial.append(mutant[d])
            else:
                trial.append(pop[i][d])
        
        # Apply weight bounds to prevent explosion
        trial = [max(-100, min(100, val)) for val in trial]
        new_pop.append(trial)
    
    return new_pop


def run_evolution(dataset: List[UserGameData], pop_size: int = 800,
                  ngen: int = 1000, cxpb: float = 0.8,
                  mutpb: float = 0.3) -> tuple:
    """
    Run Differential Evolution algorithm to find optimal feature weights
    DE/rand/1/bin strategy with self-adaptive parameters
    """
    # Initialize population
    pop = toolbox.population(n=pop_size)
    
    # Evaluate initial population
    fitnesses = []
    for ind in pop:
        fit = toolbox.evaluate(ind, dataset)
        ind.fitness.values = fit
        fitnesses.append(fit[0])
    
    # Hall of Fame for elitism
    hof = tools.HallOfFame(int(pop_size * 0.05))
    hof.update(pop)
    
    # DE parameters with self-adaptation
    F = 0.5  # Differential weight
    CR = 0.9  # Crossover rate
    F_min, F_max = 0.3, 0.9
    CR_min, CR_max = 0.5, 0.9
    
    # Evolution loop with early convergence detection
    best_fitness_history = []
    no_improvement_count = 0
    
    for gen in range(ngen):
        # Progress reporting
        if gen % 50 == 0 or gen == ngen - 1:
            best_fitness = hof[0].fitness.values[0] if hof else float('inf')
            print(f"Generation {gen+1}/{ngen}: Best fitness = {best_fitness:.4f} (F={F:.3f}, CR={CR:.3f})")
            
            # Track fitness for convergence detection
            best_fitness_history.append(best_fitness)
            
            # Check for convergence
            if len(best_fitness_history) >= 2:
                improvement = best_fitness_history[-2] - best_fitness_history[-1]
                if improvement < 0.0001:
                    no_improvement_count += 1
                    if no_improvement_count >= 2:
                        print(f"DE converged after {gen+1} generations (improvement < 0.0001 for 100 generations)")
                        break
                else:
                    no_improvement_count = 0
        
        # Self-adaptive DE parameters based on generation progress
        progress = gen / ngen
        F = F_min + (F_max - F_min) * (1 - progress)  # Start high, end low
        CR = CR_min + (CR_max - CR_min) * progress  # Start low, end high
        
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
        
        # Selection: replace current with trial if better
        new_pop = []
        new_fitnesses = []
        for i in range(len(pop)):
            if trial_fitnesses[i] < fitnesses[i]:  # Better fitness (lower is better)
                new_pop.append(trial_pop[i])
                new_fitnesses.append(trial_fitnesses[i])
            else:
                new_pop.append(pop[i])
                new_fitnesses.append(fitnesses[i])
        
        # Apply elitism: ensure best individuals survive
        hof.update(new_pop)
        
        # Replace worst individuals with elites if needed
        if len(hof) > 0:
            # Sort population by fitness
            sorted_indices = sorted(range(len(new_pop)), key=lambda i: new_fitnesses[i])
            
            # Replace worst individuals with elites
            for i, elite in enumerate(hof):
                if i < len(sorted_indices):
                    worst_idx = sorted_indices[i]
                    new_pop[worst_idx] = creator.Individual(list(elite))
                    new_pop[worst_idx].fitness.values = elite.fitness.values
                    new_fitnesses[worst_idx] = elite.fitness.values[0]
        
        pop = new_pop
        fitnesses = new_fitnesses
    
    # Return best individual
    best = hof[0]
    return best