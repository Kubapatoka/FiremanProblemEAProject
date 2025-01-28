import networkx as nx
import numpy as np
import pandas as pd
from ProblemDef import FirefighterProblem


class Instance:
    def __init__(self, file_name, evaluator, mutator, generator, fixer, weight_generator):
        self.problem: FirefighterProblem = FirefighterProblem.load_from_file(file_name)
        self._mutator   = mutator
        self._evaluator = evaluator
        self._fixer     = fixer
        self._generator = generator
        self._weight_generator = weight_generator
        self.size = len(self.problem.graph)

        self.solutions = []
        self.metrics = pd.DataFrame(columns=["iteration", "time", "min", "mean", "max", "std"])

    def new_problem(self):
        self.problem = FirefighterProblem.load_from_file(file_name)

    def mut(self, gene, o):
        ofixed = self.fix(o, distribution=gene)
        o1 = self._mutator(self.problem, gene, ofixed)
        return self.fix(o1, distribution=gene)

    def eval(self, o):
        return self._evaluator(self.problem, o)

    def gen(self):
        return self._generator(self.problem)

    def fix(self, o, distribution=None):
        res = self._fixer(self.problem, o, distribution=distribution)
        if res.sum() != self.problem.num_teams:
            print("pre:", o)
            print("post:", res)
            print()
        assert res.sum() == self.problem.num_teams
        return res

    def get_weights(self, objective_values):
        return self._weight_generator(self.problem, objective_values)
