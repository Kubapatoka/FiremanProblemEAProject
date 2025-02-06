import scipy.stats as stats
from ProblemDef import FirefighterProblem


class BetaMean:
    def __init__(self, std_dev=0.2):
        self.std_dev = std_dev

    def __call__(self, problem: FirefighterProblem):
        mean = problem.num_teams / len(problem.graph)
        variance = self.std_dev**2
        alpha = max(((1 - mean) / variance - 1 / mean) * mean**2, 1e-6)
        beta_param = max(alpha * (1 / mean - 1), 1e-6)  # Ensure beta > 0

        samples = stats.beta.rvs(alpha, beta_param, size=len(problem.graph))
        return samples


class TruncatedNormal:
    def __init__(self, scale=0.2, loc=0.5, lower=0, upper=1):
        self.scale = scale
        self.loc = loc
        self.a, self.b = (lower - self.loc) / self.scale, (
            upper - self.loc
        ) / self.scale

    def __call__(self, problem):
        return stats.truncnorm.rvs(
            self.a, self.b, loc=self.loc, scale=self.scale, size=len(problem.graph)
        )


class Beta:
    def __init__(self, mean=0.5, std_dev=0.2):
        self.std_dev = std_dev
        self.variance = self.std_dev**2
        self.mean = mean

    def __call__(self, problem: FirefighterProblem):
        alpha = max(
            ((1 - self.mean) / self.variance - 1 / self.mean) * self.mean**2, 1e-6
        )
        beta_param = max(alpha * (1 / self.mean - 1), 1e-6)  # Ensure beta > 0

        samples = stats.beta.rvs(alpha, beta_param, size=len(problem.graph))
        return samples
