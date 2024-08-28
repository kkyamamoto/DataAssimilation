import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stat
import numpy.linalg as la
import scipy.linalg as sla
import scipy.optimize as op
from model.StochasticModel import StochasticModel
from model.LotkaVolterraModel import LotkaVolterraModel
import sampling.ImportanceSampler as ImportanceSampler
import math


class MCMC:
    def __init__(self, dim=10, chainLength=100000):
        self.dim = dim
        self.chainLength = chainLength
        # self.optimalStepSize = 1 / (self.dim**2) # this is for HW 9 #3
        # self.optimalStepSize = 1 / (self.dim ** 1)  # this is for HW 9 #3
        self.optimalStepSize = 1 / np.sqrt(self.dim)

        self.chain = np.full((self.chainLength, self.dim), np.nan)
        self.acceptRatio = np.full(self.chainLength - 1, np.nan)
        self.averageAcceptRatio = None
        # self.iact

    def posteriorLotkaVolterra(self, theta):  # theta is the 6-element vector as described in Hw 9 #3
        for n in range(len(theta)):
            if theta[n] > 10:
                return 0
        d = [[2.000000000000000, 4.467000000000001], [2.000000000000000, 2.623000000000000],
             [2.200000000000000, 1.171000000000000], [2.700000000000000, 0.797000000000000],
             [5.000000000000000, 1.184000000000000], [5.500000000000000, 1.438000000000000],
             [7.800000000000001, 1.552000000000000], [7.000000000000000, 2.949000000000000],
             [5.900000000000000, 3.524000000000000], [2.800000000000000, 4.242000000000000],
             [2.000000000000000, 4.664000000000000]]
        lvm = LotkaVolterraModel(theta)
        dM = lvm.solveLotkaVolterra(np.arange(0, 11, 1))
        p = 0
        for i in range(10):
            p += la.norm(d[i] - dM[i]) ** 2
        p = np.exp(-p / 2)
        return p

    def runRWMLotkaVolterra(self, initial=None):
        if initial == None:
            currentState = np.zeros(self.dim)
        else:
            currentState = initial

        self.chain[0] = currentState
        for n in range(1, self.chainLength):
            xPrime = self.chain[n - 1] + self.optimalStepSize * np.random.multivariate_normal(np.zeros(self.dim),
                                                                                              np.mat(np.identity(
                                                                                                  self.dim)))
            a = self.posteriorLotkaVolterra(xPrime) / self.posteriorLotkaVolterra(self.chain[n - 1])
            self.acceptRatio[n - 1] = np.minimum(1, a)
            u = np.random.uniform()
            if u < self.acceptRatio[n - 1]:
                self.chain[n] = xPrime
            else:
                self.chain[n] = self.chain[n - 1]
        self.averageAcceptRatio = np.mean(self.acceptRatio)

    def runRWMGaussian(self, initial=None):
        if initial == None:
            currentState = np.zeros(self.dim)
        else:
            currentState = initial

        chain = np.full((self.chainLength, self.dim), np.nan)
        chain[0] = currentState
        for n in range(1, self.chainLength):
            xPrime = chain[n - 1] + self.optimalStepSize * np.random.multivariate_normal(np.zeros(self.dim),
                                                                                         np.mat(np.identity(self.dim)))
            # computation of a below assumes the covariance of target distribution is identity
            xPrimeMatCol = np.mat(xPrime).T
            currentMatCol = np.mat(chain[n - 1]).T
            muMatCol = np.mat(np.zeros(self.dim)).T
            a = -(xPrimeMatCol - muMatCol).T * (xPrimeMatCol - muMatCol) / 2 + (currentMatCol - muMatCol).T * (
                        currentMatCol - muMatCol) / 2
            # a=stat.multivariate_normal.pdf(xPrime, np.zeros(self.dim),np.identity(self.dim))/stat.multivariate_normal.pdf(chain[n-1], np.zeros(self.dim),np.identity(self.dim))
            a = np.exp(a)
            self.acceptRatio[n - 1] = np.minimum(1, a)
            u = np.random.uniform()
            if u < self.acceptRatio[n - 1]:
                chain[n] = xPrime
            else:
                chain[n] = chain[n - 1]
        self.averageAcceptRatio = np.mean(self.acceptRatio)

    def plotAcceptRatio(self):
        t = [10, 50, 100, 200, 500]

        averageAcceptRatioArray = np.full(len(t), np.nan)
        for n in range(len(t)):
            self.dim = t[n]
            self.runRWMGaussian()
            averageAcceptRatioArray[n] = self.averageAcceptRatio
            print("Computed for n=" + str(t[n]) + " | Ave Accept Ratio=" + str(averageAcceptRatioArray[n]))
        print(averageAcceptRatioArray)
        plt.plot(t, averageAcceptRatioArray, '-', label='Average Acceptance Ratio')
        plt.title('Average Acceptance Ratio vs Dimension')
        plt.xlabel('Dimension')
        plt.ylabel('Ave Accept Ratio')

        plt.show()

    def plotHareLynx(self):
        t = np.arange(0, 11, 1)
        lvm = LotkaVolterraModel()
        for n in range(10):
            lvm.setInitial(self.chain[np.random.randint(0, self.chainLength)])
            model = np.array(lvm.solveLotkaVolterra(t)).transpose()
            plt.plot(t, model[0], 'C0:', label='Hare Fur')
            plt.plot(t, model[1], 'C1:', label='Lynx Fur')

        d = [[2.000000000000000, 4.467000000000001], [2.000000000000000, 2.623000000000000],
             [2.200000000000000, 1.171000000000000], [2.700000000000000, 0.797000000000000],
             [5.000000000000000, 1.184000000000000], [5.500000000000000, 1.438000000000000],
             [7.800000000000001, 1.552000000000000], [7.000000000000000, 2.949000000000000],
             [5.900000000000000, 3.524000000000000], [2.800000000000000, 4.242000000000000],
             [2.000000000000000, 4.664000000000000]]

        # plt.plot(t, np.array(d).transpose()[0], 'k-', label='Hare Fur')
        # plt.plot(t, np.array(d).transpose()[1], 'k-', label='Lynx Fur')

        plt.plot(t, np.array(d).transpose()[0], 'o', label='Hare Fur')
        plt.plot(t, np.array(d).transpose()[1], 'o', label='Lynx Fur')

        plt.title('Hare and Lynx Furs')
        plt.xlabel('Year')
        plt.ylabel('Fur Amount')

        plt.show()


def main():
    # HW 9 #2
    m = MCMC(100)
    m.runRWMGaussian()
    # print(m.acceptRatio)
    print(m.averageAcceptRatio)

    # m=MCMC()
    # m.plotAcceptRatio()

    # HW 9 #3
    # m=MCMC(6,10000)
    # # initial=[0.5861,0.2345,0.7780,0.1768,2.5786,3.8248]
    # initial = [0.5861, 0.2345, 0.7780, 0.1768, 1, 5]
    # # initial = [2,2,2,2, 5, 5]
    # m.runRWMLotkaVolterra(initial)
    # m.plotHareLynx()
    # print(m.averageAcceptRatio)


if __name__ == "__main__":
    main()
