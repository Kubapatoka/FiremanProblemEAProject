import numpy as np
from scipy.spatial.distance import cdist

from .MTGAInstance import Instance
from Utils import *
import Utils

from time import time
from random import random


class MTGA:
    def __init__(self, instance: Instance, **kwargs):
        self.instance: Instance = instance

        self.mutation_probability = kwargs.get("mutation_probability", 0.3)
        self.tribe_num = kwargs.get("tribe_number", 10)
        self.tribe_pop = kwargs.get("tribe_population", 50)
        self.gene_mutation_probability = kwargs.get("gene_mutation_probability", 0.25)
        self.gene_mutation_frequency = kwargs.get("gene_mutation_frequency", 5)

        self.zero_impact = kwargs.get(
            "zero_impact", -float(instance.problem.num_teams) / instance.size
        )
        self.one_impact = kwargs.get(
            "one_impact",
            float(instance.size - instance.problem.num_teams) / instance.size,
        )

        self.local_step_size = kwargs.get("local_step_size", 0.5)

        self.global_step_size = kwargs.get("global_step_size", 0.05)
        self.global_susceptibility_radius_factor = kwargs.get(
            "global_susceptibility_radius_factor", 0.5
        )
        self.global_influence_factor = kwargs.get("global_influence_factor", 1.0)
        self.global_density_correction_function = kwargs.get(
            "global_density_correction_function", lambda x: x
        )

        self.sigmoid_steepness = kwargs.get("correction_sigmoid_steepness", 0.1)
        self.softmax_temperature = kwargs.get("correction_softmax_temperature", 0.00)
        self.smooth_clip_steepness = kwargs.get("correction_smooth_clip_steepness", 50)

        self.print_weights = kwargs.get("print_weights", False)
        self.weights_output_filename = kwargs.get(
            "weights_output_filename", "output/weights.it{iteration}.txt"
        )
        self.output_file = open("/dev/null", "w")

        self.feedback_damping_factor = kwargs.get("damping_factor", 1.0)
        self.feedback_boost_factor = kwargs.get("boost_factor", 1.0)

        if kwargs.get("correction_function", "") == "sigmoid":
            self.correction_function = lambda arr, delta: sigmoid(
                arr + delta, self.sigmoid_steepness
            )
        elif kwargs.get("correction_function", "") == "softmax":
            self.correction_function = lambda arr, delta: softmax(
                arr + delta, self.softmax_temperature
            )
        elif kwargs.get("correction_function", "") == "smooth_clip":
            self.correction_function = lambda arr, delta: Utils.smooth_clip(
                arr + delta, 0, self.smooth_clip_steepness
            )
        elif kwargs.get("correction_function", "") == "complex":
            self.correction_function = lambda arr, delta: self.dampen_feedback(
                arr, delta
            )
        else:
            self.correction_function = lambda arr, delta: np.clip(arr + delta, 0, 1)

    def initialize_genes(self):
        genes = np.zeros((self.tribe_num, self.instance.size), dtype=np.float64)
        for i in range(self.tribe_num):
            genes[i, :] = self.instance.gen()
        return genes

    def generate_population(self, genes):
        population = np.zeros(
            (self.tribe_num, self.tribe_pop, self.instance.size), dtype=bool
        )
        for i, gene in enumerate(genes):
            for j in range(self.tribe_pop):
                random_values = np.random.rand(self.instance.size)
                population[i, j, :] = self.instance.fix(
                    (random_values < gene).astype(bool), distribution=gene
                )
        self.check_population(population)
        return population

    def generate_mutated_population(self, genes):
        population = np.zeros(
            (self.tribe_num, self.tribe_pop, self.instance.size), dtype=bool
        )
        for i, gene in enumerate(genes):
            for j in range(self.tribe_pop):
                random_values = np.random.rand(self.instance.size)
                child = self.instance.fix(
                    (random_values < gene).astype(bool), distribution=gene
                )
                if random() < self.mutation_probability:
                    population[i, j, :] = self.instance.mut(gene, child)
                else:
                    population[i, j, :] = child
        self.check_population(population)
        return population

    def eval_population(self, population):
        flat_population = population.reshape(self.tribe_num * self.tribe_pop, -1)
        flat_objective_values = np.array(
            [self.instance.eval(ind) for ind in flat_population]
        )
        objective_value = flat_objective_values.reshape(self.tribe_num, self.tribe_pop)
        return objective_value

    def print_aligned(self, gene, idx, feedback, *args):
        def print_float(x):
            if x < 0:
                return f"{x:2.4f}"
            else:
                return f" {x:2.4f}"

        np.set_printoptions(
            linewidth=1000, precision=4, formatter={"float": print_float}
        )
        self.output_file.write(
            f"======================== Tribe {idx} ========================"
        )
        max_len = max([8] + [len(name) for name, _ in args])
        for name, arg in args:
            self.output_file.write(f"{name.ljust(max_len)}: {arg}")

        self.output_file.write(f"{"feedback".ljust(max_len)}: {feedback}")
        self.output_file.write(f"{"gene".ljust(max_len)}: {gene}")
        # numpy_reset_default_prints()

    def calculate_density_factor(self, genes):
        sq_dists = ((genes[:, None, :] - genes[None, :, :]) ** 2).sum(axis=2)
        K = np.exp(-sq_dists / (2 * self.global_susceptibility_radius_factor**2))
        np.fill_diagonal(K, 0)
        rho = np.sum(K, axis=1)
        return rho, K

    def calculate_local_feedback(self, genes, rho, population, weights_array):
        local_feedback = np.zeros(
            (self.tribe_num, self.instance.size), dtype=np.float64
        )

        for i in range(self.tribe_num):
            scaled_population = (
                population[i, :, :].astype(np.float64)
                * (self.one_impact - self.zero_impact)
            ) + self.zero_impact
            delta = weights_array[i, :, None] * scaled_population
            delta_sum = delta.sum(axis=0)
            local_feedback[i, :] = delta_sum * rho[i] * self.local_step_size
            # if self.print_weights:
            self.output_file.write("\nLocal Weights")
            self.print_aligned(
                genes[i, :],
                i,
                local_feedback[i, :],
                ("rho", rho),
                ("delta", delta_sum),
            )
        return local_feedback

    def calculate_global_feedback_old(self, genes, gene_values):
        gene_weights = self.instance.get_weights(gene_values)
        weighted_mean = np.sum(genes * gene_weights[:, None], axis=0) / np.sum(
            gene_weights
        )

        distances = cdist(genes, genes, metric="euclidean")
        sigma = np.mean(distances)
        affinity_matrix = np.exp(-distances / sigma)
        scores = np.sum(affinity_matrix, axis=1) - np.diag(affinity_matrix)

        inverse_weights = (1.0 - gene_weights[:, None]) * scores
        global_feedback = (
            inverse_weights * (weighted_mean - genes) * self.global_step_size
        )
        # if self.print_weights:
        self.output_file.write("\nGlobal Weights")
        for i in range(self.tribe_num):
            self.print_aligned(
                genes[i, :],
                i,
                global_feedback[i, :],
                ("gene_values", gene_values),
                ("gene_weights", gene_weights),
                ("weighted_mean", weighted_mean),
                ("scores", scores),
                ("inverse_weights", inverse_weights),
            )

        return global_feedback

    def calculate_global_feedback(self, genes, rho, K, gene_values):
        I = np.exp(self.global_influence_factor * gene_values)
        weighted_K = K * I[None, :]
        numerator = (weighted_K * genes) - genes * weighted_K.sum(axis=1)[:, None]
        denominator = weighted_K.sum(axis=1)

        update_direction = np.zeros_like(genes)
        nonzero = denominator > 1e-8
        update_direction[nonzero] = numerator[nonzero] / denominator[nonzero, None]

        S = (1 - I) * self.global_density_correction_function(rho)
        global_feedback = self.global_step_size * S[:, None] * update_direction
        return global_feedback

    def dampen_feedback(self, genes, feedback):
        deviation = genes - 0.5
        multiplier = np.ones_like(genes)

        pushing_mask = (feedback * deviation) > 0
        pulling_mask = (feedback * deviation) < 0

        multiplier[pushing_mask] = 1 - self.feedback_damping_factor * (
            2 * np.abs(deviation[pushing_mask])
        )
        multiplier[pulling_mask] = 1 + self.feedback_boost_factor * (
            2 * np.abs(deviation[pulling_mask])
        )
        multiplier = np.clip(multiplier, 0, 1)
        new_genes = genes + feedback * multiplier
        return new_genes

    def judge_population(self, genes, population, objective_value, weights_array, it):
        self.check_shapes(genes, population, objective_value, weights_array)
        if self.print_weights:
            print_sorted_parameters(
                self.output_file,
                objective_value,
                weights_array,
            )

        rho, _ = self.calculate_density_factor(genes)
        local_feedback = self.calculate_local_feedback(
            genes, rho, population, weights_array
        )
        # gene_values = self.eval_gene(objective_value, weights_array)
        # global_feedback = self.calculate_global_feedback(genes, rho, K, gene_values)

        mutation_changes = np.zeros(genes.shape)
        if it % self.gene_mutation_frequency:
            mutation_changes = self.instance.mutate_genes(genes)

        new_genes = self.correction_function(genes, local_feedback + mutation_changes)
        return new_genes

    # Now the less important functions
    def calculate_gene_weights(self, objective_value):
        return np.array(
            [
                self.instance.get_weights(objective_value[i, :])
                for i in range(self.tribe_num)
            ]
        )

    def check_shapes(self, genes, population, objective_value, weights_array):
        check_shape(genes, (self.tribe_num, self.instance.size), "genes")
        check_shape(
            population,
            (self.tribe_num, self.tribe_pop, self.instance.size),
            "population",
        )
        check_shape(
            objective_value, (self.tribe_num, self.tribe_pop), "objective_value"
        )
        check_shape(weights_array, (self.tribe_num, self.tribe_pop), "weights_array")

    def check_population(self, population):
        sums = population.sum(axis=2)
        if np.any(sums != self.instance.problem.num_teams):
            print(population[(sums != self.instance.problem.num_teams)])
            assert False, "population not valid"

    def find_best_solution(self, population, objective_value):
        flat_population = population.reshape(self.tribe_num * self.tribe_pop, -1)
        flat_objective_values = objective_value.reshape(self.tribe_num * self.tribe_pop)
        solution_idx = (
            flat_objective_values.argmax()
        )  # we are searching for highest value
        self.instance.solutions.append(
            (flat_population[solution_idx], flat_objective_values[solution_idx])
        )

    def eval_gene(self, objective_value, weights_array):
        return np.sum(weights_array * objective_value, axis=-1)

    def reset_output_file(self, iteration):
        self.output_file.close()
        self.output_file = open(
            self.weights_output_filename.format(iteration=iteration), "w"
        )


def optimize(instance: Instance, **kwargs):
    number_of_iterations = kwargs.get("number_of_iterations", 100)
    mtga = MTGA(instance, **kwargs)
    current_genes = mtga.initialize_genes()

    run_time0 = time()
    for it in range(number_of_iterations):
        mtga.reset_output_file(it)
        iter_time0 = time()
        population = mtga.generate_mutated_population(current_genes)
        objective_value = mtga.eval_population(population)
        weights_array = mtga.calculate_gene_weights(objective_value)
        current_genes = mtga.judge_population(
            current_genes, population, objective_value, weights_array, it
        )

        mtga.find_best_solution(population, objective_value)

        iter_time = time() - iter_time0
        register_metrics(
            instance.metrics,
            it,
            time() - run_time0,
            iter_time,
            objective_value,
            **kwargs,
        )


def optimize_and_collect(instance, **kwargs):
    number_of_iterations = kwargs.get("number_of_iterations", 100)
    if kwargs.get("reset_recored", True):
        instance.reset()
    mtga = MTGA(instance, **kwargs)
    current_genes = mtga.initialize_genes()

    collected_data = []  # To store data across generations

    run_time0 = time()
    for it in range(number_of_iterations):
        mtga.reset_output_file(it)
        iter_time0 = time()
        population = mtga.generate_mutated_population(current_genes)
        objective_value = mtga.eval_population(population)
        weights_array = mtga.calculate_gene_weights(objective_value)
        current_genes = mtga.judge_population(
            current_genes, population, objective_value, weights_array, it
        )

        evaluations = mtga.eval_gene(objective_value, weights_array)
        collected_data.append(
            {
                "generation": it,
                "genes": current_genes.copy(),
                "evaluations": evaluations,
            }
        )

        mtga.find_best_solution(population, objective_value)

        iter_time = time() - iter_time0
        register_metrics(
            instance.metrics,
            it,
            time() - run_time0,
            iter_time,
            objective_value,
            **kwargs,
        )

    return collected_data
