import numpy as np
import networkx as nx
from ProblemDef import FirefighterProblem

class ChoiceFixer:
    def __call__(self, problem: FirefighterProblem, o, distribution=None):
        current_true_count = o.sum()
        target_true_count = problem.num_teams

        if current_true_count > target_true_count:
            true_indices = np.where(o)[0]
            to_turn_off = np.random.choice(true_indices,
                current_true_count - target_true_count, replace=False)
            o[to_turn_off] = False
        elif current_true_count < target_true_count:
            false_indices = np.where(~o)[0]
            to_turn_on = np.random.choice(false_indices,
                target_true_count - current_true_count, replace=False)
            o[to_turn_on] = True

        return o

class WeightedChoiceFixer:
    def __call__(self, problem: FirefighterProblem, o, distribution):
        current_true_count = o.sum()
        target_true_count = problem.num_teams
        change_distribution = 1 - distribution

        if current_true_count > target_true_count:
            true_indices = np.where(o)[0]
            probabilities = change_distribution[true_indices]
            probabilities /= probabilities.sum()
            to_turn_off = np.random.choice(
                true_indices,
                current_true_count - target_true_count,
                replace=False,
                p=probabilities)
            o[to_turn_off] = False
        elif current_true_count < target_true_count:
            false_indices = np.where(~o)[0]
            probabilities = change_distribution[false_indices]
            probabilities /= probabilities.sum()  # Normalize to ensure sum is 1
            to_turn_on = np.random.choice(
                false_indices,
                target_true_count - current_true_count,
                replace=False,
                p=probabilities)
            o[to_turn_on] = True

        return o
