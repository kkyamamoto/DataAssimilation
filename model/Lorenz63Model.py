import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import ode
from scipy.integrate import odeint
import math


class Lorenz63Model:
    def __init__(self, timeIncrement=0.01, alpha=10, beta=8 / 3, rho=28):
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.noiseCov = np.mat(np.identity(3))
        self.timeIncrement = timeIncrement
        self.lorenzInitial = np.random.uniform(-0.01, 0.01, 3)

    def solveLorenz63(self, totalTimeSteps):
        x = np.full((totalTimeSteps + 1, 3), np.nan)
        initial = self.lorenzInitial
        x[0] = initial
        for n in range(totalTimeSteps):
            x[n + 1] = self.solveLorenz63OneStep(initial)
            initial = x[n + 1]
        return x

    def solveLorenz63OneStep(self, x):  # x is 3-element array of current state
        x[0] = x[0] + self.timeIncrement * self.alpha * (x[1] - x[0])
        x[1] = x[1] + self.timeIncrement * (x[0] * (self.rho - x[2]) - x[1])
        x[2] = x[2] + self.timeIncrement * (x[0] * x[1] - self.beta * x[2])

        return x + math.sqrt(self.timeIncrement) * np.random.multivariate_normal(np.zeros(3), self.noiseCov)
        # return x


def main():
    l = Lorenz63Model()
    print(l.solveLorenz63(10))


if __name__ == "__main__":
    main()
