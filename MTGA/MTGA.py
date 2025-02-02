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

    self.mut_prob = kwargs.get("mutation_probability", 0.3)
    self.tribe_num = kwargs.get("tribe_number", 10)
    self.tribe_pop = kwargs.get("tribe_population", 1000)

    self.zero_impact = kwargs.get(
      "zero_impact", instance.problem.num_teams // instance.size
    )

    self.local_feedback_weight = kwargs.get("local_feedback_weight", 0.5)
    self.global_feedback_weight = kwargs.get("global_feedback_weight", 0.05)

    self.sigmoid_steepness = kwargs.get("correction_sigmoid_steepness", 0.1)
    self.softmax_temperature = kwargs.get("correction_softmax_temperature", 0.00)
    self.smooth_clip_steepness = kwargs.get("correction_smooth_clip_steepness", 50)

    self.print_weights = kwargs.get("print_weights", False)
    self.weights_output_filename = kwargs.get(
      "weights_output_filename", "output/weights.it{iteration}.txt"
    )

    if kwargs.get("correction_function", "") == "sigmoid":
      self.correction_function = lambda arr: sigmoid(arr, self.sigmoid_steepness)
    elif kwargs.get("correction_function", "") == "softmax":
      self.correction_function = lambda arr: softmax(arr, self.softmax_temperature)
    elif kwargs.get("correction_function", "") == "smooth_clip":
      self.correction_function = lambda arr: Utils.smooth_clip(
        arr, 0, self.smooth_clip_steepness
      )
    else:
      self.correction_function = lambda arr: np.clip(arr, 0, None)

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
        if random() < self.mut_prob:
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

  def calculate_gene_weights(self, objective_value):
    return np.array(
      [self.instance.get_weights(objective_value[i, :]) for i in range(self.tribe_num)]
    )

  def judge_population(self, genes, population, objective_value, weights_array):
    check_shape(genes, (self.tribe_num, self.instance.size), "genes")
    check_shape(
      population,
      (self.tribe_num, self.tribe_pop, self.instance.size),
      "population",
    )
    check_shape(objective_value, (self.tribe_num, self.tribe_pop), "objective_value")
    check_shape(weights_array, (self.tribe_num, self.tribe_pop), "weights_array")

    local_feedback = np.zeros((self.tribe_num, self.instance.size), dtype=np.float64)
    for i in range(self.tribe_num):
      scaled_population = population[i, :, :].astype(np.float64) * self.instance.size
      delta = (
        weights_array[i, :, None]
        * (scaled_population - self.instance.problem.num_teams - 1)
        / self.instance.size
      )
      local_feedback[i, :] = delta.sum(axis=0) * self.local_feedback_weight

    # then do global chromosome feedback
    gene_values = self.eval_gene(objective_value, weights_array)
    gene_weights = self.instance.get_weights(gene_values)
    weighted_mean = np.sum(genes * gene_weights[:, None], axis=0) / np.sum(gene_weights)

    distances = cdist(genes, genes, metric="euclidean")
    inverse_weights = 1.0 - gene_weights[:, None]
    global_feedback = (
      inverse_weights * (weighted_mean - genes) * self.global_feedback_weight
    )

    new_genes = genes + local_feedback + global_feedback
    return self.correction_function(new_genes)

  def check_population(self, population):
    sums = population.sum(axis=2)
    if np.any(sums != self.instance.problem.num_teams):
      print(population[(sums != self.instance.problem.num_teams)])
      assert False, "population not valid"

  def find_best_solution(self, population, objective_value):
    flat_population = population.reshape(self.tribe_num * self.tribe_pop, -1)
    flat_objective_values = objective_value.reshape(self.tribe_num * self.tribe_pop)
    solution_idx = flat_objective_values.argmax()  # we are searching for highest value
    self.instance.solutions.append(
      (flat_population[solution_idx], flat_objective_values[solution_idx])
    )

  def eval_gene(self, objective_value, weights_array):
    return np.sum(weights_array * objective_value, axis=-1)


def print_weights(filename, objective_value, weights_array):
  sorted_indices = np.argsort(objective_value, axis=1)
  sorted_values = np.take_along_axis(objective_value, sorted_indices, axis=1)
  sorted_weights = np.take_along_axis(weights_array, sorted_indices, axis=1)

  with open(filename, "w") as file:
    i = 0
    for row_values, row_weights in zip(sorted_values, sorted_weights):
      file.write(f"Tribe {i}\n")
      i += 1
      for v, w in zip(row_values, row_weights):
        file.write(f"- {v:3.3f} {w:3.3f}\n")
  print(f"Sorted values and weights saved to {filename}")


def optimize(instance: Instance, **kwargs):
  number_of_iterations = kwargs.get("number_of_iterations", 100)
  mtga = MTGA(instance, **kwargs)
  current_genes = mtga.initialize_genes()

  run_time0 = time()
  for it in range(number_of_iterations):
    iter_time0 = time()
    population = mtga.generate_mutated_population(current_genes)
    objective_value = mtga.eval_population(population)
    weights_array = mtga.calculate_gene_weights(objective_value)
    # print("weights_array", weights_array.shape, weights_array)
    if mtga.print_weights:
      print_weights(
        mtga.weights_output_filename.format(iteration=it),
        objective_value,
        weights_array,
      )
    current_genes = mtga.judge_population(
      current_genes, population, objective_value, weights_array
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
  mtga = MTGA(instance, **kwargs)
  current_genes = mtga.initialize_genes()

  collected_data = []  # To store data across generations

  run_time0 = time()
  for it in range(number_of_iterations):
    iter_time0 = time()
    population = mtga.generate_mutated_population(current_genes)
    objective_value = mtga.eval_population(population)
    weights_array = mtga.calculate_gene_weights(objective_value)
    print_weights(f"output/weights.it{it}.txt", objective_value, weights_array)
    current_genes = mtga.judge_population(
      current_genes, population, objective_value, weights_array
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
