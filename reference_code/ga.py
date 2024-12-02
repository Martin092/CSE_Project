import numpy as np
from ase import Atoms
from ase.optimize import BFGS
from ase.calculators.lj import LennardJones
from ase.io import write, read
from ase.visualize import view


# Define the Lennard-Jones potential for carbon atoms
def lj_potential(positions):
    atoms = Atoms('C' * len(positions), positions=positions)
    calc = LennardJones()
    atoms.calc = calc  # Update to avoid deprecation warning
    return atoms.get_potential_energy()


# Create an initial population of structures
def initialize_population(size, num_atoms, bounds):
    return [np.random.uniform(bounds[0], bounds[1], (num_atoms, 3)) for _ in range(size)]


# Select parents based on fitness (lower energy is better)
def select_parents(population, fitness):
    min_energy = fitness.min()
    max_energy = fitness.max()
    fitness = (max_energy - fitness) / (max_energy - min_energy)
    total_fitness = fitness.sum()
    if total_fitness == 0:
        fitness += 1e-10  # Avoid division by zero

    selected_indices = np.random.choice(np.arange(len(population)), size=len(population) // 2,
                                        p=fitness / total_fitness)
    return [population[i] for i in selected_indices]


# Crossover operation
def crossover(parent1, parent2):
    point = np.random.randint(1, parent1.shape[0] - 1)
    child1 = np.vstack((parent1[:point], parent2[point:]))
    child2 = np.vstack((parent2[:point], parent1[point:]))
    return child1, child2


# Mutation operation
def mutate(individual, mutation_rate, bounds):
    if np.random.rand() < mutation_rate:
        individual += np.random.uniform(-0.1, 0.1, individual.shape)
        individual = np.clip(individual, bounds[0], bounds[1])  # Keep within bounds
    return individual


# Genetic Algorithm
def genetic_algorithm(pop_size, num_atoms, generations, bounds, mutation_rate):
    # Initialize population
    population = initialize_population(pop_size, num_atoms, bounds)

    best_overall_energy = float('inf')
    best_overall_structure = None

    for gen in range(generations):
        # Evaluate fitness
        fitness = np.array([lj_potential(ind) for ind in population])

        # Print current generation best fitness
        best_fitness = fitness.min()
        print(f"Generation {gen}: Best Fitness = {best_fitness}")

        # Update overall best structure if needed
        if best_fitness < best_overall_energy:
            best_overall_energy = best_fitness
            best_overall_structure = population[np.argmin(fitness)].copy()

        # Selection
        parents = select_parents(population, fitness)

        # Create new population
        new_population = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                child1, child2 = crossover(parents[i], parents[i + 1])
                new_population.extend([mutate(child1, mutation_rate, bounds), mutate(child2, mutation_rate, bounds)])

        # Fill the population if necessary
        while len(new_population) < pop_size:
            new_population.append(
                mutate(np.random.uniform(bounds[0], bounds[1], (num_atoms, 3)), mutation_rate, bounds))

        population = new_population

    # Final optimization with BFGS on the best solution
    atoms = Atoms('C' * num_atoms, positions=best_overall_structure)
    calc = LennardJones()
    atoms.calc = calc

    # Use BFGS optimizer
    optimizer = BFGS(atoms)
    optimizer.run(fmax=0.05)

    return atoms.get_positions(), atoms.get_potential_energy()

if __name__ == '__main__':

    # Parameters
    population_size = 50
    number_of_atoms = 13  # Change as needed
    generations = 150
    bounds = np.array([-2.0, 2.0])  # Change based on your system
    mutation_rate = 0.1

    np.random.seed(19)  # 19 - almost global optimal

    # Run the genetic algorithm
    optimized_positions, optimized_energy = genetic_algorithm(
        population_size, number_of_atoms, generations, bounds, mutation_rate
    )

    print("Optimized Positions:\n", optimized_positions)
    print("Optimized Energy:", optimized_energy)

    # Save the overall best structure
    write('ga.xyz', Atoms('C' + str(number_of_atoms), positions=optimized_positions))

    # Visualize the final optimized structure
    final_atoms = read('ga.xyz')
    view(final_atoms)
