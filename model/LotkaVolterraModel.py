import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import ode
from scipy.integrate import odeint


class LotkaVolterraModel:
    def __init__(self, fullParam=[0.5861, 0.2345, 0.7780, 0.1768, 2.5786, 3.8248]):
        self.initial = [fullParam[4], fullParam[5]]
        self.alpha = fullParam[0]
        self.beta = fullParam[1]
        self.gamma = fullParam[2]
        self.delta = fullParam[3]

    def setInitial(self, fullParam):
        self.initial = [fullParam[4], fullParam[5]]
        self.alpha = fullParam[0]
        self.beta = fullParam[1]
        self.gamma = fullParam[2]
        self.delta = fullParam[3]

    def LotkaVolterra(self, x, t):
        d = np.zeros(2)
        d[0] = self.alpha * x[0] - self.beta * x[0] * x[1]
        d[1] = -self.gamma * x[1] + self.delta * x[0] * x[1]

        return d

    def solveLotkaVolterra(self, timeSteps):
        r = odeint(self.LotkaVolterra, self.initial, timeSteps)
        return r


def main():
    l = LotkaVolterraModel()
    print(l.solveLotkaVolterra(np.arange(0, 11, 1)))


if __name__ == "__main__":
    main()
