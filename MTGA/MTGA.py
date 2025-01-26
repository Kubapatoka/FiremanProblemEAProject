import numpy as np
from typing import Final
from .MTGAInstance import Instance
from ..RecordKeeping import register_metrics

from time import time

class MTGA:
    def __init__(self, instance: Instance, **kwargs):
        self.instance: Instance = instance

        self.mut_prob:   Final[float] = kwargs.get('mutation_probability', 0.3)
        self.cross_prob: Final[float] = kwargs.get('crossover_probability', 0.95)
        self.tribe_num:  Final[int]   = kwargs.get('tribe number', 10)
        self.tribe_pop:  Final[int]   = kwargs.get('tribe population', 20)
        self.population_shape: Final[tuple[int]] = (self.tribe_num, self.tribe_pop, self.instance.size)

    def initialize_genes(self):
        genes = np.zeros((self.tribe_num, self.instance.size), dtype=np.float64)
        for i in range(self.tribe_num):
            genes[i, :] = instance.gen()
        return genes

    def generate_population(self, genes):
        population = np.zeros((self.tribe_num, self.tribe_pop, self.instance.size), dtype=bool)
        for i, gene in enumerate(genes):
            random_values = np.random.rand(self.tribe_pop, self.instance.size)
            population[i,:,:] = (random_values < gene).astype(bool)
        return population


    def eval_population(self, population):
        # objective_value = np.zeros((self.tribe_num, self.tribe_pop), dtype=np.float64)
        # tribe_num, tribe_pop = population.shape[:2]
        flat_population = population.reshape(self.tribe_num * self.tribe_pop, -1)
        flat_objective_values = np.array([self.instance.eval(ind) for ind in flat_population])
        objective_value = flat_objective_values.reshape(tribe_num, tribe_pop)
        return objective_value

    def mutate_population(self, genes, population):
        new_population = np.zeros(*self.population_shape, dtype=bool)
        for i, gene in enumerate(genes):
            for j in range(self.tribe_pop):
                if random.random() < self.mut_prob:
                    new_population[i,j,:] = instance.mut(gene, population[i,j,:])
                else:
                    new_population[i,j,:] = population[i,j,:]
        return new_population

    def judge_population(self, genes, population, new_population,
                         objective_value, new_objective_value):
        # do something (feedback)
        return new_genes, population, objective_value

def optimize(instance: Instance, **kwargs):
    number_of_iterations = kwargs.get('number_of_iterations', 100)
    mtga = MTGA(instance, **kwargs)
    current_genes = mtga.initialize_genes()

    run_time0 = time()
    for it in range(number_of_iterations):
        iter_time0 = time()
        population = mtga.generate_population(current_genes)
        objective_value = mtga.eval_population(population)
        mutated = mtga.mutate_population(genes, population)
        mut_objective_value = mtga.eval_population(mutated)
        current_genes, population, objective_value = mtga.judge_population(
            current_genes, population, mutated, objective_value, mut_objective_value
        )

        solution_idx = objective_value.argmax() # we are searching for highest value
        instance.solution.append(population[solution_idx], objective_value[solution_idx])

        iter_time = time() - iter_time0
        register_metrics(None, it, time() - run_time0, iter_time, objective_value, **kwargs)


def optimize_and_record(instance: Instance, **kwargs):
    number_of_iterations = kwargs.get('number_of_iterations', 100)
    mtga = MTGA(instance, **kwargs)
    current_genes = mtga.initialize_genes()

    run_time0 = time()
    for it in range(number_of_iterations):
        iter_time0 = time()
        population = mtga.generate_population(current_genes)
        objective_value = mtga.eval_population(population)
        mutated = mtga.mutate_population(genes, population)
        mut_objective_value = mtga.eval_population(mutated)
        current_genes, population, objective_value = mtga.judge_population(
            current_genes, population, mutated, objective_value, mut_objective_value
        )

        solution_idx = objective_value.argmax() # we are searching for highest value
        instance.solution.append(population[solution_idx], objective_value[solution_idx])

        iter_time = time() - iter_time0
        register_metrics(instance.metrics, it, time() - run_time0, iter_time, objective_value, **kwargs)
