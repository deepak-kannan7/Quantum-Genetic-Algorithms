import numpy as np
import matplotlib.pyplot as plt

# Define the Ackley function
def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
    cos_term = -np.exp(np.sum(np.cos(c * x) / d))
    return a + np.exp(1) + sum_sq_term + cos_term

# Define the Quantum Genetic Algorithm class
class QuantumGeneticAlgorithm:
    def __init__(self, num_variables, population_size, num_generations):
        self.num_variables = num_variables
        self.population_size = population_size
        self.num_generations = num_generations
        self.population = None
        self.best_fitness_values = []

    def initialize_population(self):
        self.population = np.random.uniform(low=-32.768, high=32.768, size=(self.population_size, self.num_variables))

    def evaluate_fitness(self):
        fitness_values = np.zeros(self.population_size)
        for i in range(self.population_size):
            fitness_values[i] = ackley(self.population[i])
        return fitness_values

    def selection(self, fitness_values):
        sorted_indices = np.argsort(fitness_values)
        sorted_population = self.population[sorted_indices]

        # Select the three best individuals
        best_individuals = sorted_population[:3]

        # Create a new population by replicating the best individuals
        self.population = np.repeat(best_individuals, repeats=self.population_size//3, axis=0)

    def crossover(self):
        new_population = np.zeros((self.population_size, self.num_variables))
        for i in range(self.population_size):
            parent1 = self.population[np.random.randint(0, self.population_size)]
            parent2 = self.population[np.random.randint(0, self.population_size)]
            crossover_point = np.random.randint(1, self.num_variables)
            new_population[i] = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        self.population = new_population

    def mutation(self):
        mutation_rate = 1 / self.num_variables
        for i in range(self.population_size):
            for j in range(self.num_variables):
                if np.random.uniform() < mutation_rate:
                    self.population[i, j] += np.random.uniform(low=-0.5, high=0.5)

    def run(self):
        self.initialize_population()

        for generation in range(self.num_generations):
            fitness_values = self.evaluate_fitness()
            self.selection(fitness_values)
            self.crossover()
            self.mutation()

            best_fitness = np.min(fitness_values)
            self.best_fitness_values.append(best_fitness)
            print(f"Generation {generation+1}: Best Fitness = {best_fitness}")
            print("Population: ")
            for i, induvidual in enumerate(self.population):
              print(f"Induvidual {i+1}: {induvidual}")
            print()

        best_solution = self.population[np.argmin(fitness_values)]
        return best_solution, best_fitness

# Usage example
num_variables = 5
population_size = 6
num_generations = 25

qga = QuantumGeneticAlgorithm(num_variables, population_size, num_generations)
best_solution, best_fitness = qga.run()

print(f"\nBest Solution: {best_solution}")
print(f"Best Fitness: {best_fitness}")

# Plotting the generation-wise best chromosome
generations = np.arange(1, num_generations+1)
best_fitness_values = qga.best_fitness_values

plt.plot(generations, best_fitness_values)
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
