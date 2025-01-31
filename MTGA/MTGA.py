import numpy as np
from typing import Final
from .MTGAInstance import Instance
from Utils import (register_metrics, check_shape)

from time import time
from random import random

class MTGA:
    def __init__(self, instance: Instance, **kwargs):
        self.instance: Instance = instance

        self.mut_prob:   Final[float] = kwargs.get('mutation_probability', 0.3)
        self.cross_prob: Final[float] = kwargs.get('crossover_probability', 0.95)
        self.tribe_num:  Final[int]   = kwargs.get('tribe number', 10)
        self.tribe_pop:  Final[int]   = kwargs.get('tribe population', 20)
        # self.local_step_size          = kwargs.get('local step size', 0.1)
        # self.global_step_size         = kwargs.get('global step size', 0.1)
        self.local_feedback_weight    = kwargs.get('local feedback weight', 0.1)
        self.global_feedback_weight   = kwargs.get('global feedback weight', 0.05)

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
        objective_value = flat_objective_values.reshape(self.tribe_num, self.tribe_pop)
        return objective_value

    def generate_mutated_population(self, genes):
        population = np.zeros((self.tribe_num, self.tribe_pop, self.instance.size), dtype=bool)
        for i, gene in enumerate(genes):
            for j in range(self.tribe_pop):
                random_values = np.random.rand(self.instance.size)
                child = self.instance.fix((random_values < gene).astype(bool), distribution=gene)
                if random() < self.mut_prob:
                    population[i,j,:] = self.instance.mut(gene, child)
        return population

    def calculate_gene_weights(self, objective_value):
        return np.array([self.instance.get_weights(objective_value[i,:]) \
            for i in range(self.tribe_num)])
        

    def judge_population(self, genes, population, objective_value, weights_array):
        check_shape(genes, (self.tribe_num, self.instance.size), "genes")
        check_shape(population, (self.tribe_num, self.tribe_pop, self.instance.size), "population")
        check_shape(objective_value, (self.tribe_num, self.tribe_pop), "objective_value")
        check_shape(weights_array, (self.tribe_num, self.tribe_pop), "weights_array")
        local_feedback = np.zeros((self.tribe_num, self.instance.size), dtype=np.float64)
        # start with per chromosome feedback
        for i in range(self.tribe_num):
            delta = weights_array[i, :, None] * ((population[i,:,:].astype(np.float64)*2)-1)
            local_feedback[i,:] = delta.sum(axis=0)

        # then do global chromosome feedback
        gene_values = self.eval_gene(objective_value, weights_array)
        gene_weights = self.instance.get_weights(gene_values)
        weighted_mean = np.sum(genes * gene_weights[:, None], axis=0) / np.sum(gene_weights)
        inverse_weights = 1.0 - gene_weights[:, None]
        global_feedback = inverse_weights * (weighted_mean - genes)

        new_genes = genes + \
            local_feedback*self.local_feedback_weight + \
            global_feedback*self.global_feedback_weight
        new_genes_pos = np.clip(new_genes, 0, None)
        new_genes_normed = new_genes_pos / np.sum(new_genes, axis=-1), 0, 1)
        return np.clip(new_genes_normed, 0, 1)

    def check_population(self, population):
        sums = arr.sum(axis=2)
        assert np.all(sums == self.instance.size) 

    def find_best_solution(self, population, objective_value):
        flat_population = population.reshape(self.tribe_num * self.tribe_pop, -1)
        flat_objective_values = objective_value.reshape(self.tribe_num * self.tribe_pop)
        solution_idx = flat_objective_values.argmax() # we are searching for highest value
        self.instance.solutions.append((flat_population[solution_idx], flat_objective_values[solution_idx]))

    def eval_gene(self, objective_value, weights_array):
        return np.sum(weights_array * objective_value, axis=-1)


def optimize(instance: Instance, **kwargs):
    number_of_iterations = kwargs.get('number of iterations', 100)
    mtga = MTGA(instance, **kwargs)
    current_genes = mtga.initialize_genes()

    run_time0 = time()
    for it in range(number_of_iterations):
        iter_time0 = time()
        population = mtga.generate_mutated_population(current_genes)
        objective_value = mtga.eval_population(population)
        weights_array = mtga.calculate_gene_weights(objective_value)
        # print("weights_array", weights_array.shape, weights_array)
        current_genes = mtga.judge_population(
            current_genes, population, objective_value, weights_array
        )

        mtga.find_best_solution(population, objective_value)

        iter_time = time() - iter_time0
        register_metrics(instance.metrics, it, time() - run_time0, iter_time, objective_value, **kwargs)

def optimize_and_collect(instance, **kwargs):
    number_of_iterations = kwargs.get('number of iterations', 100)
    mtga = MTGA(instance, **kwargs)
    current_genes = mtga.initialize_genes()

    collected_data = []  # To store data across generations

    run_time0 = time()
    for it in range(number_of_iterations):
        iter_time0 = time()
        population = mtga.generate_mutated_population(current_genes)
        objective_value = mtga.eval_population(population)
        weights_array = mtga.calculate_gene_weights(objective_value)
        current_genes = mtga.judge_population(
            current_genes, population, objective_value, weights_array
        )

        evaluations = mtga.eval_gene(objective_value, weights_array)
        collected_data.append({
            'generation': it,
            'genes': current_genes.copy(),
            'evaluations': evaluations
        })

        mtga.find_best_solution(population, objective_value)

        iter_time = time() - iter_time0
        register_metrics(instance.metrics, it, time() - run_time0, iter_time, objective_value, **kwargs)

    return collected_data
