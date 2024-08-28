import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import scipy.optimize as op
from model.StochasticModel import StochasticModel
from model.Lorenz63Model import Lorenz63Model
import sampling.ImportanceSampler as ImportanceSampler
from model.Lorenz96Model import Lorenz96Model
import math


class ParticleFilter:
    def __init__(self, nEnsembles=20, lorenzDim=None):
        self.nEnsembles = nEnsembles
        self.lorenzDim = lorenzDim
        self.timeIncrement = 1
        self.finalTime = 100
        # self.timeIncrement = 0.01
        # self.finalTime = 3
        self.totalTimeSteps = int(self.finalTime / self.timeIncrement)
        self.trueState = np.full((self.totalTimeSteps + 1, 3), np.nan)
        self.syntheticDataDim = 2
        self.syntheticData = np.full((self.totalTimeSteps + 1, self.syntheticDataDim), np.nan)
        self.initialEnsemble = np.full(self.nEnsembles, np.nan)
        # self.initialEnsemble = np.full((self.nEnsembles, 3), np.nan)
        self.R = np.mat(np.identity(self.syntheticDataDim))
        self.H = 1
        # self.H= np.mat(np.array([[1,0,0],[0,0,1]]))
        self.filterReconstructionMean = np.full((self.totalTimeSteps + 1, 3), np.nan)
        self.filterReconstructionCov = np.full((self.totalTimeSteps + 1, 3, 3), np.nan)
        self.rmse = None
        self.enkfReconstructionCovTrace = None
        self.resamplingHistogram = None

    # def generateSyntheticDataStochastic(self):
    #     stochastic=StochasticModel()
    #     simulation=stochastic.solveStochastic(self.totalTimeSteps)
    #     for n in range(len(self.syntheticData)):
    #         self.trueState[n] = simulation[n]
    #         self.syntheticData[n]=simulation[n]**2/20+np.random.normal(0,1)

    def generateSyntheticDataLorenz63(self):
        L63 = Lorenz63Model()
        initial = L63.solveLorenz63(500)
        L63.lorenzInitial = initial[len(initial) - 1]
        simulation = L63.solveLorenz63(self.totalTimeSteps)
        for n in range(len(self.syntheticData)):
            self.trueState[n] = simulation[n]
            self.syntheticData[n][0] = simulation[n][0]
            self.syntheticData[n][1] = simulation[n][2]
            self.syntheticData[n] += np.random.multivariate_normal(np.zeros(self.syntheticDataDim), self.R)

    def generateInitialEnsembleStochastic(self):
        for n in range(self.nEnsembles):
            self.initialEnsemble[n] = np.random.normal(0, 100)

    def generateInitialEnsembleLorenz63(self):
        L63 = Lorenz63Model()
        initial = L63.solveLorenz63(3000)
        L63.lorenzInitial = initial[len(initial) - 1]
        simulation = L63.solveLorenz63(5000)
        for n in range(self.nEnsembles):
            self.initialEnsemble[n] = simulation[np.floor(np.random.uniform(0, len(simulation) - 1))]

    def runStandardPFStochastic(self,
                                obsTimeSteps=1):  # obsTimeSteps=1 means obs at every model step, =2 means obs at every other model step
        # Use this for HW 9 #1
        H = self.H
        w = np.full(self.nEnsembles, np.nan)

        x = self.initialEnsemble
        # mu = np.mat(np.mean(x, axis=0)).T
        # P = np.cov(np.mat(x).T)

        # y=self.syntheticData

        # self.filterReconstructionMean[0] = mu.T
        # self.filterReconstructionCov[0] = P

        stochastic = StochasticModel()
        # L63=Lorenz63Model()

        for n in range(self.totalTimeSteps):
            for ne in range(self.nEnsembles):
                x[ne] = stochastic.solveStochasticOneStep(x[ne], n + 1)
                # x[ne]=L63.solveLorenz63OneStep(x[ne])
            if (n + 1) % obsTimeSteps == 0:
                for ne in range(self.nEnsembles):
                    y = x[ne] ** 2 / 20 + np.random.normal(0, 1)
                    w[ne] = np.exp(-(y - H * x[ne]) ** 2 / 2)

                x = ImportanceSampler.performResampling(w, x)

            self.plotResamplingHistogram(x)

            # mu = np.mat(np.mean(x, axis=0)).T
            # P = np.cov(np.mat(x).T)
            #
            # self.filterReconstructionMean[n + 1] = mu.T
            # self.filterReconstructionCov[n + 1] = P

    def runStandardPFLorenz63(self,
                              obsTimeSteps=1):  # obsTimeSteps=1 means obs at every model step, =2 means obs at every other model step
        H = self.H
        w = np.full(self.nEnsembles, np.nan)

        x = self.initialEnsemble
        mu = np.mat(np.mean(x, axis=0)).T
        P = np.cov(np.mat(x).T)

        y = self.syntheticData

        self.filterReconstructionMean[0] = mu.T
        self.filterReconstructionCov[0] = P

        L63 = Lorenz63Model()

        for n in range(self.totalTimeSteps):
            for ne in range(self.nEnsembles):
                x[ne] = L63.solveLorenz63OneStep(x[ne])
            if (n + 1) % obsTimeSteps == 0:
                for ne in range(self.nEnsembles):
                    w[ne] = np.exp(-(np.mat(y[n + 1]).T - H * np.mat(x[ne]).T).T * la.inv(self.R) * (
                                np.mat(y[n + 1]).T - H * np.mat(x[ne]).T) / 2)

                x = ImportanceSampler.performResampling3D(w, x)

            mu = np.mat(np.mean(x, axis=0)).T
            P = np.cov(np.mat(x).T)

            self.filterReconstructionMean[n + 1] = mu.T
            self.filterReconstructionCov[n + 1] = P

    # Below is standard particle filter with obs at every model step. The above general method reproduces this when obsTimeSteps=1.
    # def runStandardPF(self):
    #     H=self.H
    #     w = np.full(self.nEnsembles, np.nan)
    #
    #     x = self.initialEnsemble
    #     mu = np.mat(np.mean(x, axis=0)).T
    #     P = np.cov(np.mat(x).T)
    #
    #     y=self.syntheticData
    #
    #     self.filterReconstructionMean[0] = mu.T
    #     self.filterReconstructionCov[0] = P
    #
    #     L63=Lorenz63Model()
    #
    #     for n in range(self.totalTimeSteps):
    #         for ne in range(self.nEnsembles):
    #             x[ne]=L63.solveLorenz63OneStep(x[ne])
    #             w[ne]=np.exp(-(np.mat(y[n+1]).T-H * np.mat(x[ne]).T).T*la.inv(self.R)*(np.mat(y[n+1]).T-H * np.mat(x[ne]).T)/2)
    #
    #         x=ImportanceSampler.performResampling3D(w, x)
    #
    #         mu = np.mat(np.mean(x, axis=0)).T
    #         P = np.cov(np.mat(x).T)
    #
    #         self.filterReconstructionMean[n + 1] = mu.T
    #         self.filterReconstructionCov[n + 1] = P

    def runStandardPfL96(self,
                         obsTimeSteps=1):  # obsTimeSteps=1 means obs at every model step, =2 means obs at every other model step
        H = self.H
        w = np.full(self.nEnsembles, np.nan)

        x = self.initialEnsemble
        mu = np.mat(np.mean(x, axis=0)).T
        P = np.cov(np.mat(x).T)

        y = self.syntheticData

        self.filterReconstructionMean[0] = mu.T
        self.filterReconstructionCov[0] = P

        L96 = Lorenz96Model(self.lorenzDim)

        for n in range(self.totalTimeSteps):
            for ne in range(self.nEnsembles):
                L96.lorenzInitial = x[ne]
                x[ne] = L96.solveLorenz96Int(np.array([0, self.timeIncrement]))[1]
            if (n + 1) % obsTimeSteps == 0:
                for ne in range(self.nEnsembles):
                    w[ne] = np.exp(-(np.mat(y[n + 1]).T - H * np.mat(x[ne]).T).T * la.inv(self.R) * (
                                np.mat(y[n + 1]).T - H * np.mat(x[ne]).T) / 2)

                x = ImportanceSampler.performResampling3D(w, x)

            mu = np.mat(np.mean(x, axis=0)).T
            P = np.cov(np.mat(x).T)

            self.filterReconstructionMean[n + 1] = mu.T
            self.filterReconstructionCov[n + 1] = P

    def runOptimalPF(self):
        H = self.H
        w = np.full(self.nEnsembles, np.nan)

        x = self.initialEnsemble
        mu = np.mat(np.mean(x, axis=0)).T
        P = np.cov(np.mat(x).T)

        y = self.syntheticData

        self.filterReconstructionMean[0] = mu.T
        self.filterReconstructionCov[0] = P

        Q = np.mat(np.identity(3))
        k = Q * H.T * la.inv(H * Q * H.T + self.R)

        L63 = Lorenz63Model(self.timeIncrement)

        for n in range(self.totalTimeSteps):
            for ne in range(self.nEnsembles):
                M_kMinus1 = np.mat(L63.solveLorenz63OneStep(x[ne])).T
                x[ne] = (M_kMinus1 + k * (np.mat(y[n + 1]).T - H * M_kMinus1)).T[0]
                w[ne] = np.exp(-(np.mat(y[n + 1]).T - H * M_kMinus1).T * la.inv(H * Q * H.T + self.R) * (
                            np.mat(y[n + 1]).T - H * M_kMinus1) / 2)

            x = ImportanceSampler.performResampling3D(w, x)

            mu = np.mat(np.mean(x, axis=0)).T
            P = np.cov(np.mat(x).T)

            self.filterReconstructionMean[n + 1] = mu.T
            self.filterReconstructionCov[n + 1] = P

    def runPerturbedObsENKF(self, obsTimeSteps=1, alpha=None,
                            r=None):  # obsTimeSteps=1 means obs at every model step, =2 means obs at every other model step
        x = self.initialEnsemble
        mu = np.mat(np.mean(x, axis=0)).T
        P = np.cov(np.mat(x).T)

        H = self.H

        self.filterReconstructionMean[0] = mu.T
        self.filterReconstructionCov[0] = P

        L63 = Lorenz63Model(self.timeIncrement)
        x_f = np.full((self.nEnsembles, self.lorenzDim), np.nan)
        for n in range(self.totalTimeSteps):
            for ne in range(self.nEnsembles):
                L63.lorenzInitial = x[ne]
                x_f[ne] = L63.solveLorenz63OneStep(x[ne])

            mu_f = np.mat(np.mean(x_f, axis=0)).T

            # # Perform inflation.
            # if(alpha!=None):
            #     for ne in range(self.nEnsembles):
            #         x_f[ne]=mu_f.T+math.sqrt(1+alpha)*(x_f[ne]-mu_f.T)

            # P_f=np.cov(np.mat(x_f).T)
            P_f = np.mat(np.zeros((self.lorenzDim, self.lorenzDim)))
            for ne in range(self.nEnsembles):
                P_f += (np.mat(x_f[ne]).T - mu_f) * (np.mat(x_f[ne]).T - mu_f).T
            P_f = P_f / (self.nEnsembles - 1)

            # # Perform localization.
            # if (r!=None):
            #     P_f=np.multiply(self.constructLocalizationMatrix(r),P_f)

            if (n + 1) % obsTimeSteps == 0:
                P_farray = np.array(P_f)
                k = P_f * H.T * la.inv(H * P_f * H.T + self.R)
                # k = P_f * H.T * la.solve((H * P_f * H.T + self.R),np.mat(np.identity(20)))

                for ne in range(self.nEnsembles):
                    yTilde = np.mat(self.syntheticData[n + 1]).T + np.mat(
                        np.random.multivariate_normal(np.zeros(self.syntheticDataDim), self.R)).T
                    x[ne] = (np.mat(x_f[ne]).T + k * (yTilde - H * np.mat(x_f[ne]).T)).T  # changed from n+1
                mu = np.mat(np.mean(x, axis=0)).T  # analysis mean
                P = np.cov(np.mat(x).T)  # analysis covariance
            else:
                mu = mu_f
                P = P_f

            self.filterReconstructionMean[n + 1] = mu.T
            self.filterReconstructionCov[n + 1] = P

    # Below is ENKF with obs at every model step. The above general method reproduces this when obsTimeSteps=1.
    # def runPerturbedObsENKF(self,alpha=None,r=None):
    #     x=self.initialEnsemble
    #     mu=np.mat(np.mean(x,axis=0)).T
    #     P=np.cov(np.mat(x).T)
    #
    #     H=self.H
    #
    #     self.filterReconstructionMean[0] = mu.T
    #     self.filterReconstructionCov[0] = P
    #
    #     L63=Lorenz63Model(self.timeIncrement)
    #     x_f = np.full((self.nEnsembles, self.lorenzDim), np.nan)
    #     for n in range(self.totalTimeSteps):
    #         for ne in range(self.nEnsembles):
    #             L63.lorenzInitial = x[ne]
    #             x_f[ne] = L63.solveLorenz63OneStep(x[ne])
    #
    #         mu_f=np.mat(np.mean(x_f,axis=0)).T
    #
    #         # # Perform inflation.
    #         # if(alpha!=None):
    #         #     for ne in range(self.nEnsembles):
    #         #         x_f[ne]=mu_f.T+math.sqrt(1+alpha)*(x_f[ne]-mu_f.T)
    #
    #         # P_f=np.cov(np.mat(x_f).T)
    #         P_f=np.mat(np.zeros((self.lorenzDim,self.lorenzDim)))
    #         for ne in range(self.nEnsembles):
    #             P_f+=(np.mat(x_f[ne]).T-mu_f)*(np.mat(x_f[ne]).T-mu_f).T
    #         P_f=P_f/(self.nEnsembles-1)
    #
    #         # # Perform localization.
    #         # if (r!=None):
    #         #     P_f=np.multiply(self.constructLocalizationMatrix(r),P_f)
    #
    #         P_farray=np.array(P_f)
    #         k=P_f*H.T*la.inv(H*P_f*H.T+self.R)
    #         # k = P_f * H.T * la.solve((H * P_f * H.T + self.R),np.mat(np.identity(20)))
    #
    #         for ne in range(self.nEnsembles):
    #             yTilde=np.mat(self.syntheticData[n+1]).T+np.mat(np.random.multivariate_normal(np.zeros(self.syntheticDataDim),self.R)).T
    #             x[ne]=(np.mat(x_f[ne]).T+k*(yTilde-H*np.mat(x_f[ne]).T)).T # changed from n+1
    #         mu=np.mat(np.mean(x,axis=0)).T # analysis mean
    #         P=np.cov(np.mat(x).T) # analysis covariance
    #
    #         self.filterReconstructionMean[n+1]=mu.T
    #         self.filterReconstructionCov[n+1]=P

    def computeRMSE(self):
        self.rmse = np.zeros(self.totalTimeSteps + 1)
        for t in range(self.totalTimeSteps + 1):
            # self.rmse[t]=la.norm(np.subtract(self.trueState[t],self.enkfReconstructionMean[t]))/math.sqrt(self.lorenzDim)
            for n in range(self.lorenzDim):
                self.rmse[t] += (self.trueState[t][n] - self.filterReconstructionMean[t][n]) ** 2
            self.rmse[t] = math.sqrt(self.rmse[t] / self.lorenzDim)

    def computeReconstructCovTrace(self):
        self.reconstructionCovTrace = np.full(self.totalTimeSteps + 1, np.nan)
        for t in range(self.totalTimeSteps + 1):
            self.reconstructionCovTrace[t] = math.sqrt(np.trace(self.filterReconstructionCov[t]) / self.lorenzDim)

    def plotError(self):
        t = np.linspace(0, self.finalTime, self.totalTimeSteps + 1)

        plt.plot(t, self.rmse, '-', label='RMSE')
        plt.plot(t, self.reconstructionCovTrace, '-', label='Normalized Trace of Analysis Covariance')
        plt.title('RMSE & Spread')
        plt.xlabel('Time')
        plt.ylabel('Error')

        plt.show()

    def plotResamplingHistogram(self, x):
        # self.resamplingHistogram=np.random.normal(self.proposal[0], self.proposal[1], 5000)
        plt.hist(x, bins='auto')  # arguments are passed to np.histogram
        # plt.hist(x, bins='auto', histtype='step')  # arguments are passed to np.histogram
        # plt.hist(self.resamplingHistogram, 30)  # arguments are passed to np.histogram
        plt.title("Resampling Histogram")
        plt.show()


def main():
    # HW 8 #1
    # p=ParticleFilter(500,3)
    # p.generateSyntheticDataLorenz63()
    # p.generateInitialEnsembleLorenz63()
    # p.runStandardPF(10)
    # # p.runOptimalPF()
    # # p.runPerturbedObsENKF(10)
    # p.computeRMSE()
    # p.computeReconstructCovTrace()
    # p.plotError()

    # HW 9 #1
    p = ParticleFilter(1000)
    p.generateInitialEnsembleStochastic()
    p.runStandardPFStochastic()


if __name__ == "__main__":
    main()
