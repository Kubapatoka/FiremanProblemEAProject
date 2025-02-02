from scipy.stats import beta
from ProblemDef import FirefighterProblem


class Generator:
    def __init__(self, std_dev=0.2):
        self.std_dev = std_dev

    def __call__(self, problem: FirefighterProblem):
        mean = problem.num_teams / len(problem.graph)
        variance = self.std_dev**2
        alpha = max(((1 - mean) / variance - 1 / mean) * mean**2, 1e-6)
        beta_param = max(alpha * (1 / mean - 1), 1e-6)  # Ensure beta > 0

        samples = beta.rvs(alpha, beta_param, size=len(problem.graph))
        return samples
