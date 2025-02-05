import scipy.stats as stats
import numpy as np
from ProblemDef import FirefighterProblem


class BetaMean:
    def __init__(self, std_dev=0.2):
        self.std_dev = std_dev

    def __call__(self, problem: FirefighterProblem, shape):
        mean = problem.num_teams / len(problem.graph)
        variance = self.std_dev**2
        alpha = max(((1 - mean) / variance - 1 / mean) * mean**2, 1e-6)
        beta_param = max(alpha * (1 / mean - 1), 1e-6)  # Ensure beta > 0

        samples = stats.beta.rvs(alpha, beta_param, size=shape)
        return samples


class TruncatedNormal:
    def __init__(self, scale=0.2, loc=0.5, lower=-0.25, upper=0.25):
        self.a = (lower - self.loc) / self.scale
        self.b = (upper - self.loc) / self.scale
        self.scale = scale
        self.loc = loc

    def __call__(self, shape):
        return stats.truncnorm.rvs(
            self.a, self.b, loc=self.loc, scale=self.scale, size=shape
        )


class Normal:
    def __init__(self, mutation_probability=0.5, sigma=0.01, mean=0.0):
        self.mut_prob = mutation_probability
        self.sigma = sigma
        self.mean = mean

    def __call__(self, problem, genes):
        tribe_num, chromosome_len = genes.shape
        changes = np.random.normal(self.mean, self.sigma, genes.shape)
        samples = np.random.random(tribe_num)
        changes[(samples > self.mut_prob)] = 0
        return changes
