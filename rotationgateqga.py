import math
import numpy as np
import matplotlib.pyplot as plt

# Define the fitness function
def fitness(x):
    return 20 + math.exp(1) - 20*math.exp(0.2*np.sum(np.square(x)/100)) - math.exp(math.cos(np.sum(2*math.pi*x))/100)

def quantum_genetic_algorithm(population_size, chromosome_length, num_generations, crossover_probability, mutation_probability):

    # Initialize the population
    population = np.random.rand(population_size, chromosome_length)

    # Initialize the fitness values
    fitness_values = np.zeros(population_size)

    for generation in range(num_generations):
        print("Population",generation,": ",population)
        # Evaluate the fitness of each individual in the population
        for i in range(population_size):
            fitness_values[i] = fitness(population[i])

        # Normalize the fitness values
        fitness_sum = np.sum(fitness_values)
        normalized_fitness_values = fitness_values / fitness_sum

        # Apply the Quantum Rotation Gate to each individual
        for i in range(population_size):
            angle = 2 * np.arccos(np.sqrt(normalized_fitness_values[i]))
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            population[i] = np.dot(rotation_matrix, population[i])

        for i in range(population_size):
            # Crossover
            if np.random.rand() < crossover_probability:
                j = np.random.randint(population_size)
                crossover_point = np.random.randint(chromosome_length)
                population[i, crossover_point:] = population[j, crossover_point:]
            # Mutation
            if np.random.rand() < mutation_probability:
                mutation_point = np.random.randint(chromosome_length)
                population[i, mutation_point] = np.random.rand()
 
    # Return the best individual and its fitness value
    best_fitness = np.inf
    best_individual = None
    for i in range(population_size):
        current_fitness = fitness(population[i])
        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_individual = population[i]

    return best_individual, best_fitness,i

# Set the parameters
population_size = 6
chromosome_length = 2
num_generations = 100
crossover_probability = 0.8
mutation_probability = 0.1

# Run the Quantum Genetic Algorithm
best_individual, best_fitness,i = quantum_genetic_algorithm(population_size, chromosome_length, num_generations, crossover_probability, mutation_probability)

# Print the results
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)
print("Best population:",i)

# Plot the results
x = np.linspace(0, 1, 40)
y = np.linspace(0, 1, 40)
X, Y = np.meshgrid(x, y)
Z = np.zeros((40, 40))
for i in range(40):
    for j in range(40):
        Z[i, j] = fitness(np.array([X[i, j], Y[i, j]]))
fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='jet')
ax.scatter(best_individual[0], best_individual[1], best_fitness, color='red', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Fitness')
plt.show()
