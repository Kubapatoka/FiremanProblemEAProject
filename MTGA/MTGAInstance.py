import networkx as nx
import numpy as np
from ProblemDef import FirefighterProblem


class Instance:
    def __init__(self, file_name, evaluator, mutator, crossover, generator, fixer):
        self.problem: FirefighterProblem = FirefighterProblem.load_from_file(file_name)
        self._mutator   = mutator
        self._crossover = crossover
        self._evaluator = evaluator
        self._fixer     = fixer
        self._generator = generator
        self.size = len(self.problem.graph)


    def mut(self, gene, o):
        o1 = self._mutator(self.problem, gene, o)
        return self.fix(o1, distribution=(1-gene))

    def eval(self, o):
        return self._evaluator(self.problem, o)

    def gen(self):
        return self._generator(self.problem)

    def fix(self, o, distribution=None):
        return self._fixer(self.problem, o, distribution=distribution)


