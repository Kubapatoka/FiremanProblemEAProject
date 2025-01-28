import numpy as np
import networkx as nx
from ProblemDef import FirefighterProblem

class WalkMutator:
    def __init__(self, number_of_walkers=4, max_length_of_walks=2):
        self.number_of_walkers = number_of_walkers
        self.max_length_of_walks = max_length_of_walks

    def select_indices(self, gene, o):
        valid_indices = np.where(o)[0]
        inverted_gene = 1 - gene
        probabilities = inverted_gene[valid_indices]
        if probabilities.sum() == 0:
            probabilities = np.ones_like(probabilities, dtype=np.float64) / len(probabilities)
        else:
            probabilities /= probabilities.sum()
        if probabilities.sum() < 1e-6:
            print(probabilities.sum(), probabilities)
            print(gene)
            print(o)

        selected_indices = np.random.choice(
            valid_indices,
            size=min(self.number_of_walkers, len(valid_indices)),
            replace=False,
            p=probabilities
        )
        return selected_indices

    def __call__(self, problem: FirefighterProblem, gene, o):
        selected_inds = self.select_indices(gene, o)
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
