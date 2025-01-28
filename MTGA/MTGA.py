import numpy as np
from typing import Final
from .MTGAInstance import Instance
from Utils import register_metrics

from time import time
from random import random

class MTGA:
    def __init__(self, instance: Instance, **kwargs):
        self.instance: Instance = instance

        self.mut_prob:   Final[float] = kwargs.get('mutation_probability', 0.3)
        self.cross_prob: Final[float] = kwargs.get('crossover_probability', 0.95)
        self.tribe_num:  Final[int]   = kwargs.get('tribe number', 10)
        self.tribe_pop:  Final[int]   = kwargs.get('tribe population', 20)
        self.local_step_size          = kwargs.get('local step size', 0.1)
        self.global_step_size         = kwargs.get('global step size', 0.1)
        self.local_feedback_weight    = kwargs.get('local feedback weight', 0.5)
        self.global_feedback_weight   = kwargs.get('global feedback weight', 0.5)

    def initialize_genes(self):
        genes = np.zeros((self.tribe_num, self.instance.size), dtype=np.float64)
        for i in range(self.tribe_num):
            genes[i, :] = self.instance.gen()
        return genes

    def generate_population(self, genes):
        population = np.zeros((self.tribe_num, self.tribe_pop, self.instance.size), dtype=bool)
        for i, gene in enumerate(genes):
            random_values = np.random.rand(self.tribe_pop, self.instance.size)
            population[i,:,:] = (random_values < gene).astype(bool)
        return population


    def eval_population(self, population):
        flat_population = population.reshape(self.tribe_num * self.tribe_pop, -1)
        flat_objective_values = np.array([self.instance.eval(ind) for ind in flat_population])
        objective_value = flat_objective_values.reshape(tribe_num, tribe_pop)
        return objective_value

    def generate_mutated_population(self, genes):
        population = np.zeros((self.tribe_num, self.tribe_pop, self.instance.size), dtype=bool)
        for i, gene in enumerate(genes):
            random_values = np.random.rand(self.tribe_pop, self.instance.size)
            population[i,:,:] = (random_values < gene).astype(bool)
            for j in range(self.tribe_pop):
                if random() < self.mut_prob:
                    population[i,j,:] = self.instance.mut(gene, population[i,j,:])
        return population

    def judge_population(self, genes, population, objective_value):
        weights_array = np.zeros((self.tribe_num, self.tribe_pop), dtype=np.float64)
        local_feedback = np.zeros((self.tribe_num, self.instance.size), dtype=np.float64)
        # start with per chromosome feedback
        for i in range(self.tribe_num):
            weights_array[i,:] = self.instance.get_weights(objective_value[i,:])
            delta = weights_array[i, :, None] * ((population[i,:,:].astype(np.float64)*2)-1)
            local_feedback[i,:] = genes[i] + self.local_step_size*delta


        # then do global chromosome feedback
        global_feedback = np.zeros((self.tribe_num, self.instance.size), dtype=np.float64)
        gene_values = np.sum(weights_array * objective_value, axis=1)
        gene_weights = self.instance.get_weights(gene_values)
        weighted_mean = np.sum(genes * gene_weights[:, None], axis=0) / np.sum(gene_weights)
        inverse_weights = 1.0 - gene_weights[:, None]
        global_feedback = self.global_step_size * inverse_weights * (weighted_mean - genes)

        new_genes = genes + \
            local_feedback*self.local_feedback_weight + \
            global_feedback*self.global_feedback_weight
        return new_genes

def optimize(instance: Instance, **kwargs):
    number_of_iterations = kwargs.get('number of iterations', 100)
    mtga = MTGA(instance, **kwargs)
    current_genes = mtga.initialize_genes()

    run_time0 = time()
    for it in range(number_of_iterations):
        iter_time0 = time()
        population = mtga.generate_mutated_population(current_genes)
        objective_value = mtga.eval_population(population)
        current_genes = mtga.judge_population(current_genes, population, objective_value)

        solution_idx = objective_value.argmax() # we are searching for highest value
        instance.solutions.append((population[solution_idx], objective_value[solution_idx]))

        iter_time = time() - iter_time0
        register_metrics(None, it, time() - run_time0, iter_time, objective_value, **kwargs)


def optimize_and_record(instance: Instance, **kwargs):
    number_of_iterations = kwargs.get('number of iterations', 100)
    mtga = MTGA(instance, **kwargs)
    current_genes = mtga.initialize_genes()

    run_time0 = time()
    for it in range(number_of_iterations):
        iter_time0 = time()
        population = mtga.generate_mutated_population(current_genes)
        objective_value = mtga.eval_population(population)
        current_genes = mtga.judge_population(current_genes, population, objective_value)

        solution_idx = objective_value.argmax() # we are searching for highest value
        instance.solutions.append((population[solution_idx], objective_value[solution_idx]))

        iter_time = time() - iter_time0
        register_metrics(instance.metrics, it, time() - run_time0, iter_time, objective_value, **kwargs)
