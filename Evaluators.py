from ProblemDef import FirefighterProblem
import networkx as nx
import numpy as np

class BaseEvaluator:
    def propagate_fire(self, problem, on_fire, burnt, firefighters):
        neighbors = set.union(*(set(problem.graph.neighbors(v)) for v in on_fire))
        burnt |= on_fire
        return neighbors - burnt - firefighters

class MainEvaluator(BaseEvaluator):
    def __call__(self, problem: FirefighterProblem, o):
        burnt = set()
        on_fire = set(problem.fire_starts)
        firefighters = set(np.flatnonzero(o))
        default = set(problem.graph.nodes) - on_fire - firefighters

        while on_fire:
            on_fire = self.propagate_fire(problem, on_fire, burnt, firefighters)
            default -= on_fire

        return len(default)

class CummulativeEvaluator(BaseEvaluator):
    def __call__(self, problem: FirefighterProblem, o):
        burnt = set()
        on_fire = set(problem.fire_starts)
        firefighters = set(np.flatnonzero(o))
        default = set(problem.graph.nodes) - on_fire - firefighters
        value = 0

        while on_fire:
            on_fire = self.propagate_fire(problem, on_fire, burnt, firefighters)
            default -= on_fire
            value += len(default)

        return value
