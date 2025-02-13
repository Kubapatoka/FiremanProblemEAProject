from ProblemDef import FirefighterProblem
from ProblemDef2 import IncrementalFirefighterProblem
import networkx as nx
import numpy as np


class BaseEvaluator:
    def propagate_fire(self, problem, on_fire, burnt):
        neighbors = set.union(*(set(problem.graph.neighbors(v)) for v in on_fire))
        burnt |= on_fire
        return neighbors - burnt


class MainEvaluator(BaseEvaluator):
    def __call__(self, problem: FirefighterProblem, o):
        burnt = set(np.flatnonzero(o))
        on_fire = set(problem.fire_starts)
        default = set(problem.graph.nodes) - on_fire - burnt

        while on_fire:
            on_fire = self.propagate_fire(problem, on_fire, burnt)
            default -= on_fire

        return len(default)


class CummulativeEvaluator(BaseEvaluator):
    def __call__(self, problem: FirefighterProblem, o):
        burnt = set(np.flatnonzero(o))
        on_fire = set(problem.fire_starts)
        default = set(problem.graph.nodes) - on_fire - burnt
        value = 0

        for _ in range(len(problem.graph.nodes)):
            if on_fire:
                on_fire = self.propagate_fire(problem, on_fire, burnt)
                default -= on_fire
            value += len(default)

        return value

class BaseIncrementalEvaluator(BaseEvaluator):
    def pick_k(self, perm : list[int], k, unavailable):
        # valid = [i for i in range(len(perm)) if perm[i] not in unavailable]
        # return perm[valid[:k]]
        return list(filter(lambda x: x not in unavailable, perm))[:k]
        

class IncrementalMainEvaluator(BaseIncrementalEvaluator):
    def __call__(self, problem: IncrementalFirefighterProblem, fireman):
        on_fire = set(problem.fire_starts)
        unavailable = set(self.pick_k(fireman, problem.num_teams_start, on_fire)) | on_fire
        default = set(problem.graph.nodes) - unavailable

        while on_fire:
            new_firefighters = set(self.pick_k(fireman, problem.num_teams_increment, unavailable))
            unavailable |= new_firefighters
            on_fire = self.propagate_fire(problem, on_fire, unavailable)
            default = default - on_fire - new_firefighters

        return len(default)

class IncrementalCummulativeEvaluator(BaseIncrementalEvaluator):
    def __call__(self, problem: IncrementalFirefighterProblem, fireman):
        on_fire = set(problem.fire_starts)
        unavailable = set(self.pick_k(fireman, problem.num_teams_start, on_fire)) | on_fire
        default = set(problem.graph.nodes) - unavailable
        value = 0

        for _ in range(len(problem.graph)):
            if on_fire:
                new_firefighters = set(self.pick_k(fireman, problem.num_teams_increment, unavailable))
                unavailable |= new_firefighters
                on_fire = self.propagate_fire(problem, on_fire, unavailable)
                default = default - on_fire - new_firefighters
            value += len(default)

        return value

