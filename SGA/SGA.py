import numpy as np
from tqdm.notebook import tqdm



class GeneticAlgo:
    def __init__(self, instance, population_size, number_of_offspring,
                 crossover_probability, mutation_probability):
        self.instance = instance

        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability

        self.population_size = population_size
        self.number_of_offspring = number_of_offspring

    def generate_population(self):
        current_population = np.zeros((self.population_size, self.instance.size), dtype=np.)
        for i in range(population_size):
            current_population[i, :] = self.instance.gen()
        return current_population

    def eval_obj_fun(self, current_population):
        objective_values = np.zeros(self.population_size)
        for i in tqdm(range(self.population_size), leave=False, position=2, desc="Evaluating Population"):
            objective_values[i] = self.instance.fit(current_population[i])
        return objective_values

    def select_parents(self, objective_values):
        fitness_values = objective_values.max() - objective_values
        if fitness_values.sum() > 0:
            fitness_values = fitness_values / fitness_values.sum()
        else:
            fitness_values = np.ones(self.population_size) / self.population_size
        return np.random.choice(self.population_size, self.number_of_offspring, True, fitness_values).astype(np.int64)

    def select_parents2(self, objective_values):
        fitness_values = objective_values.max() - objective_values
        exp_fitness = np.exp(fitness_values - np.max(fitness_values))
        probabilities = exp_fitness / np.sum(exp_fitness)

        indices = np.zeros(self.number_of_offspring)
        for i in tqdm(range(len(objective_values)//2), leave=False, position=2, desc="Selecting parents"):
            indices[i], indices[i+1] = np.random.choice(len(objective_values), size=2, replace=False, p=probabilities)

        if len(indices) % 2 == 1:
            indices[-1] = np.random.choice(len(objective_values), size=1, p=probabilities)
        return indices



    # creating the children population
    def create_children(self, parent_indices, current_population):
        children_population = [ None ] * self.number_of_offspring
        for i in tqdm(range(int(self.number_of_offspring/2)), leave=False, position=2, desc="Creating Children"):
            parent1 = copy.deepcopy(current_population[parent_indices[2*i]])
            parent2 = copy.deepcopy(current_population[parent_indices[2*i+1]])
            if np.random.random() < self.crossover_probability:
                child1, child2 = self.instance.cross2(parent1, parent2)
                children_population[2*i], children_population[2*i+1] = child1, child2
            else:
                children_population[2*i], children_population[2*i+1] = parent1, parent2
        if np.mod(self.number_of_offspring, 2) == 1:
            children_population[-1] = current_population[parent_indices[-1]]
        return children_population

    # mutating the children population
    def mutate_children(self, children_population):
        for i in tqdm(range(self.number_of_offspring), leave=False, position=2, desc="Mutating Children"):
            if np.random.random() < self.prune_mutation_probability or children_population[i].depth > 8:
                children_population[i] = self.instance.prune_mut(children_population[i])
            elif np.random.random() < self.subst_mutation_probability:
                children_population[i] = self.instance.subst_mut(children_population[i])
            elif np.random.random() < self.op_swap_mutation_probability:
                children_population[i] = self.instance.op_swap_mut(children_population[i])

    def replace_population(self, current_population, children_population, objective_values, children_objective_values):
        objective_values = np.hstack([objective_values, children_objective_values])
        current_population = current_population + children_population

        sorted_indices = np.argsort(objective_values)
        selected_indices = sorted_indices[:self.population_size]

        current_population = [ current_population[i] if i < self.population_size else children_population[i-self.population_size] for i in selected_indices ]
        objective_values = objective_values[selected_indices]

        current_population = [current_population[i] if 

        return current_population, objective_values

