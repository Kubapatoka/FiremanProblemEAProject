import numpy as np
import networkx as nx
from ProblemDef import FirefighterProblem
from Utils import softmax

import math

class WalkMutator:
    def __init__(self, ratio_of_workers=0.1, max_length_of_walks=2):
        self.ratio_of_workers = ratio_of_workers
        self.max_length_of_walks = max_length_of_walks

    def select_indices(self, num_teams, gene, o):
        number_of_walkers = int(math.ceil(self.ratio_of_workers * num_teams))
        valid_indices = np.where(o)[0]
        inverted_gene = 1 - gene
        probabilities = inverted_gene[valid_indices]
        if probabilities.sum() == 0:
            probabilities = np.ones_like(probabilities, dtype=np.float64) / len(probabilities)
        else:
            probabilities /= probabilities.sum()
        if np.abs(probabilities.sum()-1) > 1e-6:
            print("probabilities dont sum up to 1")
            print(probabilities.sum(), probabilities)
            print(gene)
            print(o)
        if np.any((probabilities < 0) | (probabilities > 1)):
            print("probabilities outside of range (0,1)")
            print(probabilities)

        if np.sum((probabilities > 1e-6)) < number_of_walkers:
            print("Fewer non-zero entries in p than size")
            print(np.sum((probabilities > 0)), number_of_walkers)
            print(probabilities)

        if probabilities.shape != valid_indices.shape:
            print("difference in shapes")
            print(probabilities.shape, valid_indices.shape)

        try:
            selected_indices = np.random.choice(
                valid_indices,
                size=min(number_of_walkers, len(valid_indices)),
                replace=False,
                p=probabilities
            )
        except ValueError as e:
            print("ValueError caught", e)
            print(probabilities)
            print(valid_indices)
            selected_indices = np.random.choice(
                valid_indices,
                size=min(number_of_walkers, len(valid_indices)),
                replace=False
            )
        except TypeError as e:
            print("TypeError caught", e)
            print(probabilities)
            print(valid_indices)
            selected_indices = np.random.choice(
                valid_indices,
                size=min(number_of_walkers, len(valid_indices)),
                replace=False
            )

        return selected_indices

    def __call__(self, problem: FirefighterProblem, gene, o):
        selected_inds = self.select_indices(problem.num_teams, gene, o)
        new_o = o.copy()
        new_o[selected_inds] = False
        for index in selected_inds:
            current_node = index

            # Perform the random walk
            for _ in range(self.max_length_of_walks):
                neighbors = [n for n in problem.graph.neighbors(current_node) if not new_o[n]]
                if not neighbors:
                    break
                next_node = np.random.choice(neighbors)
                current_node = next_node

            new_o[current_node] = True

        return new_o
