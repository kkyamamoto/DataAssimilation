import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import scipy.optimize as op
import scipy.stats as stat
import math


class ImportanceSampler:
    def __init__(self, target=[0, 4], proposal=[0, 1], distribDim=None, epsilon=None):
        # target and proposal arrays contain the mean and (co)variance of a normal distribution
        self.distribDim = distribDim  # if distribDim != None (and computeRhoMultiDim is run), then the target and proposal arguments are ignored -- see computeRhoMultiDim method for how target and proposal distributions are defined
        self.epsilon = epsilon
        self.target = [target[0], math.sqrt(target[
                                                1])]  # if computeRhoBimodal method is run, then the target argument is ignored -- see computeRhoBimodal method for the bimodal distribution used to define the target
        self.proposal = [proposal[0], math.sqrt(proposal[1])]

        self.resamplingHistogram = None

    def function(self, x):
        if (x >= 4):
            return 1
        else:
            return 0

    def computeMcEstimator(self, nSamples):
        x = np.random.normal(self.proposal[0], self.proposal[1], nSamples)
        f_p_over_q = np.full(nSamples, np.nan)

        for i in range(len(x)):
            f_p_over_q[i] = self.function(x[i]) * stat.norm.pdf(x[i], self.target[0], self.target[1]) / stat.norm.pdf(
                x[i], self.proposal[0], self.proposal[1])

        return np.mean(f_p_over_q)

    def computeAverageError(self, nSamples, trueExpectation, nExperiments):
        error = np.full(nExperiments, np.nan)

        for i in range(len(error)):
            estimate = self.computeMcEstimator(nSamples)
            error[i] = abs(trueExpectation - estimate) / abs(trueExpectation)

        return np.mean(error)

    def computeRhoMultiDim(self, nSamples):
        x = np.random.multivariate_normal(np.zeros(self.distribDim), (1 + self.epsilon) * np.identity(self.distribDim),
                                          nSamples)
        w = np.full(nSamples, np.nan)
        wSquared = np.full(nSamples, np.nan)

        for i in range(len(x)):
            w[i] = stat.multivariate_normal.pdf(x[i], np.zeros(self.distribDim),
                                                np.identity(self.distribDim)) / stat.multivariate_normal.pdf(x[i],
                                                                                                             np.zeros(
                                                                                                                 self.distribDim),
                                                                                                             (
                                                                                                             1 + self.epsilon) * np.identity(
                                                                                                                 self.distribDim))
            wSquared[i] = w[i] * w[i]

        wMean = np.mean(w)
        return np.mean(wSquared) / (wMean * wMean)

    def computeRhoBimodal(self, nSamples):
        x = np.random.normal(self.proposal[0], self.proposal[1], nSamples)
        # x = 0.7*np.random.normal(0,1, nSamples)+0.3*np.random.normal(4,1, nSamples)
        # x = np.random.uniform(-2,6,nSamples)
        w = np.full(nSamples, np.nan)
        wSquared = np.full(nSamples, np.nan)

        for i in range(len(x)):
            w[i] = (0.7 * stat.norm.pdf(x[i], 0, 1) + 0.3 * stat.norm.pdf(x[i], 4, 1)) / stat.norm.pdf(x[i],
                                                                                                       self.proposal[0],
                                                                                                       self.proposal[1])
            # w[i] = stat.norm.pdf(x[i],self.target[0],self.target[1]) / stat.norm.pdf(x[i],self.proposal[0],self.proposal[1])
            wSquared[i] = w[i] * w[i]

        self.resamplingHistogram = self.performResampling(w, x)

        wMean = np.mean(w)
        return np.mean(wSquared) / (wMean * wMean)

    def computeRhoBimodalAverage(self, nSamples, nExperiments):
        rho = np.full(nExperiments, np.nan)

        for i in range(len(rho)):
            rho[i] = self.computeRhoBimodal(nSamples)

        return np.mean(rho)

    def performResampling(self, unnormalizedWeights, originalSamples):
        nSamples = len(unnormalizedWeights)
        normalizedW = unnormalizedWeights / np.sum(unnormalizedWeights)
        # normalizedW=unnormalizedWeights
        histogram = np.full(nSamples, np.nan)
        occurrence = np.zeros(nSamples)
        cdf = np.full(nSamples, np.nan)

        sum = 0
        for i in range(nSamples):
            sum += normalizedW[i]
            cdf[i] = sum

        uniform = np.random.uniform(0, 1, nSamples)
        # uniform = np.random.uniform(0, np.sum(unnormalizedWeights), nSamples)

        for i in range(nSamples):
            n = 0
            while uniform[i] >= cdf[n]:
                n += 1

            histogram[i] = originalSamples[n]
            occurrence[n] += 1

        return histogram

    def plotResamplingHistogram(self):
        # self.resamplingHistogram=np.random.normal(self.proposal[0], self.proposal[1], 5000)
        plt.hist(self.resamplingHistogram, bins='auto')  # arguments are passed to np.histogram
        # plt.hist(self.resamplingHistogram, 30)  # arguments are passed to np.histogram
        plt.title("Resampling Histogram with Proposal [" + str(self.proposal[0]) + ", " + str(
            self.proposal[1] * self.proposal[1]) + "]")
        plt.show()


def main():
    # # HW 5 #1 and #2
    # trueExpectation=0.0227501 # This is for HW 5 #1 and #2, where p ~ N(0,4) and f is as given in assignment. See https://goo.gl/CL6sYR
    #
    # i=ImportanceSampler([0,4],[2,1])
    # # estimate=i.computeMcEstimator(100000)
    # # print(estimate)
    # # print(abs(trueExpectation - estimate) / abs(trueExpectation))
    # print(i.computeAverageError(10000,trueExpectation,10))

    # # HW 5 #3
    # # print(np.random.normal(0,1,5))
    # # print(np.random.multivariate_normal(np.zeros(1),(1+0)*np.identity(1),3))
    #
    # i=ImportanceSampler([0,1],[0,1],470,0.1)
    # print(i.computeRhoMultiDim(100000))

    # HW 5 #4 -- This is the plot of target p: https://goo.gl/BBmwt5
    i = ImportanceSampler([0, 1], [2, 5])
    print(i.computeRhoBimodal(50000))
    # # print(i.computeRhoBimodalAverage(1000,100))
    i.plotResamplingHistogram()

    # unnormal=np.full(10,10.0)
    # print(unnormal)
    # print(unnormal/2)

    # print("Histogram Proposal ["+str(i.proposal[0])+", "+str(i.proposal[1])+"]")


if __name__ == "__main__":
    main()
