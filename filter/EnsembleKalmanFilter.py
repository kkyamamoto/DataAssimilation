import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
from diffEqModel.Lorenz96Model import Lorenz96Model
import math

class EnsembleKalmanFilter:
    def __init__(self,nEnsembles=20,lorenzDim=40):
        self.nEnsembles=nEnsembles
        self.lorenzDim=lorenzDim
        self.timeIncrement=0.1
        self.finalTime=5
        self.totalTimeSteps=int(self.finalTime/self.timeIncrement)
        self.trueState = np.full((self.totalTimeSteps + 1, self.lorenzDim), np.nan)
        self.syntheticDataDim=int(self.lorenzDim/2)
        self.syntheticData=np.full((self.totalTimeSteps+1,self.syntheticDataDim),np.nan)
        self.initialEnsemble=np.full((self.nEnsembles,self.lorenzDim),np.nan)
        self.R = np.mat(np.identity(self.syntheticDataDim))
        self.enkfReconstructionMean = np.full((self.totalTimeSteps + 1, self.lorenzDim), np.nan)
        self.enkfReconstructionCov = np.full((self.totalTimeSteps + 1, self.lorenzDim, self.lorenzDim), np.nan)
        self.rmse=None
        self.enkfReconstructionCovTrace=None

        self.optimalInflationAlpha=None
        self.optimalLocalizationR=None

    # def generateSyntheticDataLorenz96RK4(self):
    #     L96=Lorenz96Model(self.lorenzDim)
    #     initial=L96.solveLorenz96RK4(100)
    #     L96.lorenzInitial=initial
    #     for n in range(len(self.syntheticData)):
    #         for m in range(self.syntheticDataDim):
    #             self.syntheticData[n][m]=L96.solveLorenz96RK4(n*0.1)[1+2*m]+np.random.normal(0,1)

    def generateSyntheticDataLorenz96Int(self):
        L96=Lorenz96Model(self.lorenzDim)
        initial=L96.solveLorenz96Int(np.arange(0,101,self.timeIncrement))
        L96.lorenzInitial = initial[len(initial)-1]
        simulation=L96.solveLorenz96Int(np.arange(0,self.finalTime+self.timeIncrement,self.timeIncrement))
        for n in range(len(self.syntheticData)):
            self.trueState[n] = simulation[n]
            for m in range(self.syntheticDataDim):
                # self.syntheticData[n][m]=simulation[n][1+2*m]+np.random.normal(0,1)
                self.syntheticData[n][m] = simulation[n][1 + 2 * m] # noise for synthetic data is added below
            self.syntheticData[n]+=np.random.multivariate_normal(np.zeros(self.syntheticDataDim),self.R)

    # def generateInitialEnsembleLorenz96RK4(self):
    #     L96=Lorenz96Model(self.lorenzDim)
    #     initial=L96.solveLorenz96RK4(1000)
    #     L96.lorenzInitial=initial
    #     for n in range(self.nEnsembles):
    #         self.initialEnsemble[n]=L96.solveLorenz96RK4(np.random.uniform(0,5000))

    def generateInitialEnsembleLorenz96Int(self):
        L96=Lorenz96Model(self.lorenzDim)
        initial=L96.solveLorenz96Int(np.arange(0,101,self.timeIncrement))
        L96.lorenzInitial=initial[len(initial)-1]
        simulation=L96.solveLorenz96Int(np.arange(0,501,self.timeIncrement))
        for n in range(self.nEnsembles):
            self.initialEnsemble[n]=simulation[np.floor(np.random.uniform(0,len(simulation)-1))]

    def runPerturbedObsENKF(self,alpha=None,r=None):
        H=np.zeros((self.syntheticDataDim,self.lorenzDim))
        for n in range(self.syntheticDataDim):
            H[n][1+2*n]=1
        H=np.mat(H)

        x=self.initialEnsemble
        mu=np.mat(np.mean(x,axis=0)).T
        P=np.cov(np.mat(x).T)

        self.enkfReconstructionMean[0] = mu.T
        self.enkfReconstructionCov[0] = P

        L96=Lorenz96Model(self.lorenzDim)
        x_f = np.full((self.nEnsembles, self.lorenzDim), np.nan)
        for n in range(self.totalTimeSteps):
            for ne in range(self.nEnsembles):
                L96.lorenzInitial = x[ne]
                x_f[ne] = L96.solveLorenz96Int(np.array([0,self.timeIncrement]))[1]

            mu_f=np.mat(np.mean(x_f,axis=0)).T

            # Perform inflation.
            if(alpha!=None):
                for ne in range(self.nEnsembles):
                    x_f[ne]=mu_f.T+math.sqrt(1+alpha)*(x_f[ne]-mu_f.T)

            # P_f=np.cov(np.mat(x_f).T)
            P_f=np.mat(np.zeros((self.lorenzDim,self.lorenzDim)))
            for ne in range(self.nEnsembles):
                P_f+=(np.mat(x_f[ne]).T-mu_f)*(np.mat(x_f[ne]).T-mu_f).T
            P_f=P_f/(self.nEnsembles-1)

            # Perform localization.
            if (r!=None):
                P_f=np.multiply(self.constructLocalizationMatrix(r),P_f)

            P_farray=np.array(P_f)
            k=P_f*H.T*la.inv(H*P_f*H.T+self.R)
            # k = P_f * H.T * la.solve((H * P_f * H.T + self.R),np.mat(np.identity(20)))

            for ne in range(self.nEnsembles):
                yTilde=np.mat(self.syntheticData[n+1]).T+np.mat(np.random.multivariate_normal(np.zeros(self.syntheticDataDim),self.R)).T
                x[ne]=(np.mat(x_f[ne]).T+k*(yTilde-H*np.mat(x_f[ne]).T)).T # changed from n+1
            mu=np.mat(np.mean(x,axis=0)).T # analysis mean
            P=np.cov(np.mat(x).T) # analysis covariance

            self.enkfReconstructionMean[n+1]=mu.T
            self.enkfReconstructionCov[n+1]=P

    def runSquareRootENKF(self,alpha=None,r=None):
        H=np.zeros((self.syntheticDataDim,self.lorenzDim))
        for n in range(self.syntheticDataDim):
            H[n][1+2*n]=1
        H=np.mat(H)

        x=self.initialEnsemble
        mu=np.mat(np.mean(x,axis=0)).T
        P=np.cov(np.mat(x).T)

        self.enkfReconstructionMean[0] = mu.T
        self.enkfReconstructionCov[0] = P

        L96=Lorenz96Model(self.lorenzDim)
        x_f = np.full((self.nEnsembles, self.lorenzDim), np.nan)
        for n in range(self.totalTimeSteps):
            for ne in range(self.nEnsembles):
                L96.lorenzInitial = x[ne]
                x_f[ne] = L96.solveLorenz96Int(np.array([0,self.timeIncrement]))[1]

            mu_f = np.mat(np.mean(x_f, axis=0)).T

            # Perform inflation.
            if(alpha!=None):
                for ne in range(self.nEnsembles):
                    x_f[ne]=mu_f.T+math.sqrt(1+alpha)*(x_f[ne]-mu_f.T)

            Xtranspose=np.full((self.nEnsembles,self.lorenzDim),np.nan)
            for ne in range(self.nEnsembles):
                Xtranspose[ne]=np.mat(x_f[ne])-mu_f.T

            X=np.mat(Xtranspose/math.sqrt(self.nEnsembles-1)).T
            P_f=X*X.T

            # Perform localization.
            if (r != None):
                P_f = np.multiply(self.constructLocalizationMatrix(r), P_f)

            P_farray = np.array(P_f)

            k = P_f * H.T * la.inv(H * P_f * H.T + self.R)

            mu=mu_f+k*(np.mat(self.syntheticData[n+1]).T-H*mu_f) # analysis mean
            P=(np.identity(self.lorenzDim)-k*H)*P_f # analysis covariance

            V=X.T*H.T
            Z=sla.sqrtm(np.mat(np.identity(self.nEnsembles))-V*la.inv(V.T*V+self.R)*V.T)

            X_a=X*Z

            for ne in range(self.nEnsembles):
                x[ne]=(mu+math.sqrt(self.nEnsembles-1)*X_a[:,ne]).T

            self.enkfReconstructionMean[n+1]=mu.T
            self.enkfReconstructionCov[n+1]=P

    def constructLocalizationMatrix(self,r):
        L=np.zeros((self.lorenzDim, self.lorenzDim))
        for i in range(self.lorenzDim):
            for j in range(i,self.lorenzDim):
                dist=min(abs(i-j),-abs(i-j)%self.lorenzDim)
                L[i][j]=np.exp(-(dist/r)**2)
        L=np.mat(L)
        L=L+L.T-np.mat(np.diag(np.diag(L)))
        Larray = np.array(L)
        return L

    def computeRMSE(self):
        self.rmse=np.zeros(self.totalTimeSteps + 1)
        for t in range(self.totalTimeSteps+1):
            # self.rmse[t]=la.norm(np.subtract(self.trueState[t],self.enkfReconstructionMean[t]))/math.sqrt(self.lorenzDim)
            for n in range(self.lorenzDim):
                self.rmse[t]+=(self.trueState[t][n]-self.enkfReconstructionMean[t][n])**2
            self.rmse[t]=math.sqrt(self.rmse[t]/self.lorenzDim)

    def computeTimeAveragedRMSE(self): # run only after running computeRMSE (to calculate self.rmse first)
        spinUpIndex=int(self.totalTimeSteps/2)
        stableRMSE=np.full(len(self.rmse)-spinUpIndex,np.nan)
        for t in range(len(stableRMSE)):
            stableRMSE[t]=self.rmse[spinUpIndex+t]
        return np.mean(stableRMSE)

    def computeEnkfReconstructCovTrace(self):
        self.kfReconstructionCovTrace = np.full(self.totalTimeSteps+1,np.nan)
        for t in range(self.totalTimeSteps + 1):
            self.kfReconstructionCovTrace[t]=math.sqrt(np.trace(self.enkfReconstructionCov[t])/self.lorenzDim)

    def plotError(self):
        t = np.linspace(0, self.finalTime, self.totalTimeSteps + 1)

        plt.plot(t, self.rmse, '-', label='RMSE')
        plt.plot(t, self.kfReconstructionCovTrace, '-', label='Normalized Trace of Analysis Covariance')
        plt.title('RMSE & Spread')
        plt.xlabel('Time')
        plt.ylabel('Error')

        plt.show()

    def optimizeInflationLocalization(self):
        timeAveragedRMSE=10
        # for alpha in np.arange(0.05,0.5,0.05):
        #     for r in np.arange(2,3,0.1):
        for alpha in np.arange(0.25, 1, 0.25):
            for r in np.arange(1, 4, 0.5):
                # self.runPerturbedObsENKF(alpha,r) # switch between perturbed-obs and square-root EnKF
                self.runSquareRootENKF(alpha, r)
                self.computeRMSE()
                timeAveragedRMSE=self.computeTimeAveragedRMSE()
                print("Computed for alpha = "+str(alpha)+" and r = "+str(r)+": RMSE = "+str(timeAveragedRMSE))

def main():
    e=EnsembleKalmanFilter(20,400)
    e.generateSyntheticDataLorenz96Int()
    e.generateInitialEnsembleLorenz96Int()
    # e.runPerturbedObsENKF()
    e.runPerturbedObsENKF(0.25,3.4)
    # e.runSquareRootENKF()
    # e.runSquareRootENKF(0.4, 3.4)
    # e.constructLocalizationMatrix(4)
    # print("Square Root EnKF")
    # e.optimizeInflationLocalization()
    e.computeRMSE()
    e.computeEnkfReconstructCovTrace()
    # timeAveragedRMSE=e.computeTimeAveragedRMSE()
    # print(timeAveragedRMSE)
    e.plotError()

if __name__ == "__main__":
    main()

