import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import scipy.optimize as op
import scipy.stats as stat
import math
import sampling.ImportanceSampler as imp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


class ImportanceSamplerHW7:
    def __init__(self, prior=None):
        # the prior array (e.g., [1,1]) contains the mean and (co)variance of a normal distribution
        if prior != None:
            self.prior = [prior[0], math.sqrt(prior[1])]
        else:
            self.prior = None

        self.resamplingHistogram = None

    def function(self, x, y=2.5):
        return (x[0] - 1) ** 2 / 2 + ((y - x[0] ** 3) / 0.1) ** 2 / 2

    def derivFunction(self, x, y=2.5):  # differentiated wrt x -- see https://goo.gl/KxLR9J
        return 300 * x ** 5 - 300 * x ** 2 * y + x - 1

    def argminF(self, y=2.5):
        return op.minimize(self.function, np.array([0]), args=(
        y,)).x  # add comma after y for a tuple with single element (although it works without it too) -- see https://goo.gl/sww5CH

    def argminFall(self, y=2.5):
        return op.minimize(self.function, np.array([0]), args=(
        y,))  # add comma after y for a tuple with single element (although it works without it too) -- see https://goo.gl/sww5CH

    def hessianF(self, x, y=2.5):  # see https://goo.gl/5M5g1h
        return 900 * x ** 4 - 600 * (y - x ** 3) * x + 1

    def function2(self, theta):  # this is function for F(theta) in #2 of HW 7
        thetaVec = np.mat([[theta[0]], [theta[1]]])  # thetaVec must be a *column* vector
        return 1e-2 * la.norm(thetaVec - np.mat([[5], [5]])) ** 4 + 0.2 * np.sin(5 * la.norm(thetaVec))

    def argminF2(self):
        return op.minimize(self.function2, np.array([0,
                                                     0])).x  # since function2 has many local minima, the argmin returned is sensitive to the initial condition

    # def invHessianF2(self):
    #     jac=np.mat(op.minimize(self.function2,np.array([0,0])).jac)
    #     return np.mat(jac.T*jac)

    def invHessianF2(self):
        return op.minimize(self.function2, np.array([0, 0])).hess_inv

    def gaussianSampler(self, nSamples, proposal=[0, 1]):
        # the proposal array (e.g., [0,1]) contains the mean and (co)variance (inverse Hessian) of a normal/t distribution
        x = np.random.normal(proposal[0], math.sqrt(proposal[1]), nSamples)
        w = np.full(nSamples, np.nan)

        for i in range(len(x)):
            w[i] = np.exp(-self.function([x[i]])) / stat.norm.pdf(x[i], proposal[0], math.sqrt(proposal[1]))

        normalizedW = w / np.sum(w)

        return [x, normalizedW]

    def gaussianSamplerMulti(self, nSamples, proposal=None):
        # the proposal array (e.g., [0,1]) contains the mean and (co)variance (inverse Hessian) of a normal/t distribution
        if proposal == None:
            proposalBody = [np.zeros(0), np.identity(2)]
        else:
            proposalBody = proposal
        # x = np.random.multivariate_normal(proposalBody[0], proposalBody[1],nSamples)
        x = np.random.multivariate_normal(proposalBody[0], math.sqrt(9) * np.identity(2), nSamples)
        # x = np.random.multivariate_normal(proposalBody[0], sla.sqrtm(np.mat(proposalBody[1])), nSamples)
        w = np.full(nSamples, np.nan)

        for i in range(len(x)):
            # w[i]=np.exp(-self.function2(x[i]))/stat.multivariate_normal.pdf(x[i], proposalBody[0], proposalBody[1])
            w[i] = np.exp(-self.function2(x[i])) / stat.multivariate_normal.pdf(x[i], proposalBody[0],
                                                                                math.sqrt(9) * np.identity(2))
            # w[i] = np.exp(-self.function2(x[i])) / stat.multivariate_normal.pdf(x[i], proposalBody[0],sla.sqrtm(np.mat(proposalBody[1])))
        normalizedW = w / np.sum(w)

        self.resamplingHistogram = self.performResampling2D(w, x)

        return [x, normalizedW]

    def tDistributionSampler(self, nSamples, nu, proposal=[0, 1]):
        # the proposal array (e.g., [0,1]) contains the mean and (co)variance (inverse Hessian) of a normal/t distribution
        sigma = math.sqrt(proposal[1] * (
                    nu - 2) / nu)  # sigma is the scale parameter in non-standardized t distribution -- see https://goo.gl/mJdvWP
        x = stat.t.rvs(nu, loc=proposal[0], scale=sigma, size=nSamples)
        w = np.full(nSamples, np.nan)

        for i in range(len(x)):
            w[i] = np.exp(-self.function([x[i]])) / stat.t.pdf(x[i], nu, loc=proposal[0], scale=sigma)

        normalizedW = w / np.sum(w)

        return [x, normalizedW]

    def randomMapSampler(self, nSamples, mu, phi, invHess):  # mu=argmin(F) and phi=min(F) see lecture p. 52 (back)
        ksi = np.random.normal(0, 1, nSamples)
        L = math.sqrt(invHess)
        x = np.full(nSamples, np.nan)
        w = np.full(nSamples, np.nan)

        for i in range(len(ksi)):
            lamb = op.newton(self.algebraicEq, ksi[i], args=(mu, phi, ksi[i], L))  # is this initial guess good?
            x = mu + lamb * L * ksi[i]
            w[i] = np.abs(lamb + 2 * ksi[i] * ksi[i] / (2 * self.derivFunction(mu + lamb * L * ksi[i]) * L * ksi[i]))

        normalizedW = w / np.sum(w)

        return [x, normalizedW]

    def algebraicEq(self, lamb, mu, phi, ksi, L):
        return self.function(mu + lamb * L * ksi) - phi - ksi * ksi / 2

    def gaussianSamplerUnnormalized(self, nSamples, proposal=[0,
                                                              1]):  # results in same quantity computed by sampler method above with normalized weights
        # the proposal array (e.g., [0,1]) contains the mean and (co)variance (inverse Hessian) of a normal/t distribution
        x = np.random.normal(proposal[0], math.sqrt(proposal[1]), nSamples)
        w = np.full(nSamples, np.nan)

        for i in range(len(x)):
            w[i] = np.exp(-self.function([x[i]])) / stat.norm.pdf(x[i], proposal[0], math.sqrt(proposal[1]))

        normalizedW = w

        return [x, normalizedW]

    def computeEffectiveSampleSize(self, normalizedW, nSamples):
        rho = np.mean(np.square(normalizedW)) / (np.mean(normalizedW) * np.mean(
            normalizedW))  # i presume this quantity is the same regardless of whether weights w are normalized
        return [rho, nSamples / rho]
        # rhoUnnormalized=np.mean(np.square(w))/(np.mean(w)*np.mean(w))
        # return [nSamples/rho,nSamples/rhoUnnormalized]

    def performResampling2D(self, unnormalizedWeights,
                            originalSamples):  # originalSamples should be a 2-dimensional array
        nSamples = len(unnormalizedWeights)
        histogram = np.full((2, nSamples), np.nan)

        originalSamplesTranspose = np.mat(originalSamples).T
        histogram[0] = imp.performResampling2D(unnormalizedWeights, np.array(originalSamplesTranspose[0][0]))
        histogram[1] = imp.performResampling2D(unnormalizedWeights, np.array(originalSamplesTranspose[1][0]))

        return histogram

    def plotTarget2(self):  # plots the target distribution for #2 in HW 7
        X = np.arange(0, 10, 0.25)
        Y = np.arange(0, 10, 0.25)
        X, Y = np.meshgrid(X, Y)
        Z = np.full((len(X), len(Y)), np.nan)

        for i in range(len(X)):
            for j in range(len(Y)):
                Z[i][j] = np.exp(-self.function2([X[i][j], Y[i][j]]))

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    def plotResamplingHistogram(self):
        # self.resamplingHistogram=np.random.normal(self.proposal[0], self.proposal[1], 5000)
        plt.hist2d(self.resamplingHistogram[0], self.resamplingHistogram[1],
                   bins=100)  # arguments are passed to np.histogram
        # plt.hist(self.resamplingHistogram, 30)  # arguments are passed to np.histogram
        plt.title("Resampling Histogram")
        plt.show()


def main():
    i = ImportanceSamplerHW7([1, 1])
    samples = 50000
    # mu=i.argminF()
    # variance=i.hessianF(mu)
    mu = i.argminF2()
    variance = i.invHessianF2()
    # print(mu)
    # print(variance)
    # print(1/variance)
    # print(i.argminFall())

    #
    # print("Gaussian proposal")
    # weights=i.gaussianSampler(samples, [mu, 1/variance])
    # effectiveSample=i.computeEffectiveSampleSize(weights[1],samples)
    # print("rho = "+str(effectiveSample[0])+" | effective samples = "+str(effectiveSample[1]))
    #
    # print("t-distribution proposal with nu=3")
    # weights = i.tDistributionSampler(samples, 3,[mu, 1 / variance])
    # effectiveSample = i.computeEffectiveSampleSize(weights[1], samples)
    # print("rho = " + str(effectiveSample[0]) + " | effective samples = " + str(effectiveSample[1]))
    #
    # print("t-distribution proposal with nu=5")
    # weights = i.tDistributionSampler(samples, 5, [mu, 1 / variance])
    # effectiveSample = i.computeEffectiveSampleSize(weights[1], samples)
    # print("rho = " + str(effectiveSample[0]) + " | effective samples = " + str(effectiveSample[1]))
    #
    # print("Random map proposal")
    # weights = i.randomMapSampler(samples, mu,i.function(mu),1/variance)
    # effectiveSample = i.computeEffectiveSampleSize(weights[1], samples)
    # print("rho = " + str(effectiveSample[0]) + " | effective samples = " + str(effectiveSample[1]))

    # #2 below
    print("Gaussian proposal")
    # i.plotTarget2()
    weights = i.gaussianSamplerMulti(samples, [mu, variance])
    effectiveSample = i.computeEffectiveSampleSize(weights[1], samples)
    print("rho = " + str(effectiveSample[0]) + " | effective samples = " + str(effectiveSample[1]))
    i.plotResamplingHistogram()


if __name__ == "__main__":
    main()
