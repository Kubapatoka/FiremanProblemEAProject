import numpy as np
import networkx as nx
from ..ProblemDef import FirefighterProblem

class WalkMutator:
    def __init__(self, number_of_walkers, max_length_of_walks):
        self.number_of_walkers = number_of_walkers
        self.max_length_of_walks = max_length_of_walks

    def select_indices(self, gene, o):
        inverted_gene = (1 - gene) * o
        normalized_probabilities = inverted_gene / inverted_gene.sum()
        valid_indices = np.where(o)[0]
        selected_indices = np.random.choice(
            valid_indices,
            size=min(self.number_of_walkers, len(valid_indices)),
            replace=False,
            p=normalized_probabilities[valid_indices]
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
