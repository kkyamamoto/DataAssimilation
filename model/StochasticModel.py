import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import ode
from scipy.integrate import odeint
import math


class StochasticModel:
    def __init__(self, timeIncrement=0.01, alpha=10, beta=8 / 3, rho=28):
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.noiseCov = np.mat(np.identity(3))
        self.timeIncrement = timeIncrement
        self.stochasticInitial = np.random.normal(0, 100)

    def solveStochastic(self, k):  # k is the time step number
        x = np.full(k + 1, np.nan)
        initial = self.stochasticInitial
        x[0] = initial
        for n in range(k):
            x[n + 1] = self.solveStochasticOneStep(initial, n + 1)
            initial = x[n + 1]
        return x

    def solveStochasticOneStep(self, x, k):  # x is scalar of current state
        return x / 2 + 25 * x / (1 + x ** 2) + 8 * np.cos(1.2 * k) + np.random.normal(0, 100)


def main():
    l = StochasticModel()
    print(l.solveStochastic(10))


if __name__ == "__main__":
    main()
