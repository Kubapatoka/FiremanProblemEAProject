from scipy.stats import beta
from ..ProblemDef import FirefighterProblem

class Generator:
    def __init__(self, std_dev):
        self.std_dev = std_dev

    def __call__(self, problem: FirefighterProblem):
        mean = problem.num_teams / len(problem.graph)
        variance = self.std_dev**2
        alpha = ((1 - mean) / variance - 1 / mean) * mean**2
        beta_param = alpha * (1 / mean - 1)

        samples = beta.rvs(alpha, beta_param, size=size)
        return samples
