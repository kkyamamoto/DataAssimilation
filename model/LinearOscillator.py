import matplotlib.pyplot as plt
import numpy as np


class LinearOscillator:
    def __init__(self, kfInitialMean=0, noiseCov=5, finalTime=100, timeIncrement=1e-3, omega=1, xi=1e-3,
                 modelInitial=[1, 2]):
        self.kfInitialMean = kfInitialMean
        self.noiseCov = noiseCov
        self.omega = omega
        self.xi = xi
        self.timeIncrement = timeIncrement
        self.modelInitial = np.array(modelInitial)
        self.finalTime = finalTime

        self.totalTimeSteps = int(self.finalTime / self.timeIncrement)

        self.A = np.array([[0, 1], [-self.omega * self.omega, -2 * self.xi * self.omega]])
        self.A2= np.array([[0, 1], [-self.omega * self.omega*1.2*1.2, -2 * self.xi * .8*self.omega*1.2]])
        self.M = np.mat(np.identity(2) + self.timeIncrement * self.A)
        self.M2=np.mat(np.identity(2) + self.timeIncrement * self.A2)

        self.trueState = None
        self.syntheticPositionData = None
        self.kfReconstructionMean = None
        self.kfReconstructionCov = None
        self.kalmanGain = None
        self.rmse=None
        self.kfReconstructionCovTrace=None

        self.computeExplicitEuler()
        self.generateSyntheticPositionData()
        self.runKalmanFilter(self.M)
        self.computeRMSE()
        self.computeKfReconstructCovTrace()

    def computeExplicitEuler(self):
        self.trueState = np.full((self.totalTimeSteps + 1, 2), np.nan)
        self.trueState[0] = self.modelInitial
        x = np.mat(self.modelInitial).T
        for n in range(self.totalTimeSteps):
            x = self.M * x
            self.trueState[n + 1] = x.T

    def generateSyntheticPositionData(self):
        self.syntheticPositionData = np.add(self.trueState.transpose()[0],
                                            np.random.normal(0, self.noiseCov, self.totalTimeSteps + 1))

    def runKalmanFilter(self,M):
        H = np.mat(np.array([1, 0]))
        mu = self.kfInitialMean * np.mat(np.ones(2)).T
        P = np.mat(np.identity(2))

        self.kfReconstructionMean = np.full((self.totalTimeSteps + 1, 2), np.nan)
        self.kfReconstructionCov = np.full((self.totalTimeSteps + 1, 2, 2), np.nan)
        self.kalmanGain = np.full((self.totalTimeSteps + 1, 2), np.nan)
        self.kfReconstructionMean[0] = mu.T
        self.kfReconstructionCov[0] = P
        self.kalmanGain[0] = np.zeros(2)

        for n in range(self.totalTimeSteps):
            mu_f = M * mu
            P_f = M * P * M.T

            k = P_f * H.T / (H * P_f * H.T + self.noiseCov)

            mu = mu_f + k * (self.syntheticPositionData[n + 1] - H * mu_f)
            P = (np.mat(np.identity(2)) - k * H) * P_f

            self.kfReconstructionMean[n + 1] = mu.T
            self.kfReconstructionCov[n + 1] = P
            self.kalmanGain[n + 1] = k.T

    def computeRMSE(self):
        self.rmse=np.full(self.totalTimeSteps + 1, np.nan)
        for n in range(self.totalTimeSteps+1):
            self.rmse[n]= ((self.trueState[n][0] - self.kfReconstructionMean[n][0]) ** 2 + (self.trueState[n][1] - self.kfReconstructionMean[n][1]) ** 2) / 2

    def computeKfReconstructCovTrace(self):
        self.kfReconstructionCovTrace = np.full(self.totalTimeSteps+1,np.nan)
        for n in range(self.totalTimeSteps + 1):
            self.kfReconstructionCovTrace[n]=np.trace(self.kfReconstructionCov[n])/2

    def plotTrueStateAndData(self):
        t = np.linspace(0, self.totalTimeSteps * self.timeIncrement, self.totalTimeSteps + 1)

        trueState = self.trueState.transpose()
        kfReconstructionMean = self.kfReconstructionMean.transpose()
        kalmanGain=self.kalmanGain.transpose()

        plt.figure(1)
        plt.subplot(2, 1, 1)
        plt.plot(t, trueState[0], '-', label='True Position')
        # plt.plot(t, self.syntheticPositionData, ',',label='Synthetic Data')
        plt.plot(t, kfReconstructionMean[0], '-', label='KF Reconstruction')
        plt.title('True Position, Synthetic Data, & KF Reconstruction')
        plt.xlabel('Time')
        plt.ylabel('Position')

        plt.subplot(2, 1, 2)
        plt.plot(t, trueState[1], '-', label='True Velocity')
        plt.plot(t, kfReconstructionMean[1], '-', label='KF Reconstruction')
        plt.title('True Velocity & KF Reconstruction')
        plt.xlabel('Time')
        plt.ylabel('Velocity')

        plt.tight_layout()

        plt.figure(2)
        plt.subplot(2, 1, 1)
        plt.plot(t, kalmanGain[0], '-', label='Position Kalman Gain')
        plt.plot(t, kalmanGain[1], '-', label='Velocity Kalman Gain')
        plt.title('Kalman Gain')
        plt.xlabel('Time')
        plt.ylabel('Kalman Gain')

        plt.subplot(2, 1, 2)
        plt.plot(t, self.rmse, '-', label='RMSE')
        plt.plot(t, self.kfReconstructionCovTrace, '-', label='Normalized Trace of Analysis Covariance')
        plt.title('RMSE & Trace of Analysis Cov')
        plt.xlabel('Time')
        plt.ylabel('Error')

        plt.tight_layout()
        plt.show()


def main():
    l = LinearOscillator()
    l.plotTrueStateAndData()


if __name__ == "__main__":
    main()
