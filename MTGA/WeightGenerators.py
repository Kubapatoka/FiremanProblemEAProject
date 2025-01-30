import numpy as np


class SoftmaxWeights:
    def __init__(self, temperature=100.0):
        self.temperature = temperature

    def __call__(self, problem, objective_values):
        max_value = np.max(objective_values)
        scaled_values = (objective_values - max_value) / self.temperature
        exp_values = np.exp(scaled_values)
        weights = exp_values / np.sum(exp_values)
        return weights


class PowerWeights:
    def __init__(self, power=2.0):
        self.power = power

    def __call__(self, problem, objective_values):
        powered_values = objective_values ** self.power
        weights = powered_values / np.sum(powered_values)
        return weights


class RankWeights:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, problem, objective_values):
        ranks = np.argsort(np.argsort(-objective_values)) + 1  # Rank 1 is the best
        weights = 1 / (ranks ** self.alpha)
        weights /= np.sum(weights)
        return weights


class SigmoidWeights:
    def __init__(self, midpoint=0.5, steepness=1.0):
        self.midpoint = midpoint
        self.steepness = steepness

    def __call__(self, problem, objective_values):
        sigmoid_values = 1 / (1 + np.exp(-self.steepness * (objective_values - self.midpoint)))
        weights = sigmoid_values / np.sum(sigmoid_values)
        return weights


class ExponentialDecayWeights:
    def __init__(self, beta=1.0, reference=None):
        self.beta = beta
        self.reference = reference  # If None, will use the max of the objective_values

    def __call__(self, problem, objective_values):
        reference = self.reference if self.reference is not None else np.max(objective_values)
        scaled_values = -self.beta * (reference - objective_values)
        exp_values = np.exp(scaled_values)
        weights = exp_values / np.sum(exp_values)
        return weights


class HuberWeights:
    def __init__(self, delta=1.0):
        self.delta = delta

    def __call__(self, problem, objective_values):
        median = np.median(objective_values)
        deviations = np.abs(objective_values - median)
        squared_loss = (objective_values - median) ** 2
        linear_loss = self.delta * deviations - 0.5 * self.delta ** 2

        # Apply Huber formula
        huber_loss = np.where(deviations <= self.delta, squared_loss, linear_loss)
        weights = huber_loss / np.sum(huber_loss)
        return weights

