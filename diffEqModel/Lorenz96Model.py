import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import ode
from scipy.integrate import odeint

class Lorenz96Model:
    def __init__(self, lorenzDim=40, lorenzForcing=8, lorenzInitial=None):
        self.lorenzDim=lorenzDim
        self.lorenzForcing=lorenzForcing
        # self.lorenzInitial=self.lorenzForcing*np.ones(self.lorenzDim)
        # self.lorenzInitial[30]+=0.01
        self.lorenzInitial = np.random.uniform(self.lorenzForcing-0.01, self.lorenzForcing+0.01, self.lorenzDim)


    def Lorenz96Int(self,x,t):
        # compute state derivatives
        N=self.lorenzDim
        F=self.lorenzForcing
        d = np.zeros(N)

        # first the 3 edge cases: i=1,2,N
        d[0] = (x[1] - x[N - 2]) * x[N - 1] - x[0]
        d[1] = (x[2] - x[N - 1]) * x[0] - x[1]
        d[N - 1] = (x[0] - x[N - 3]) * x[N - 2] - x[N - 1]

        # then the general case
        for i in range(2, N - 1):
            d[i] = (x[i + 1] - x[i - 2]) * x[i - 1] - x[i]

        # add the forcing term
        d = d + F

        # return the state derivatives
        return d

    def Lorenz96RK4(self,t,x):
        # compute state derivatives
        N=self.lorenzDim
        F=self.lorenzForcing
        d = np.zeros(N)

        # first the 3 edge cases: i=1,2,N
        d[0] = (x[1] - x[N - 2]) * x[N - 1] - x[0]
        d[1] = (x[2] - x[N - 1]) * x[0] - x[1]
        d[N - 1] = (x[0] - x[N - 3]) * x[N - 2] - x[N - 1]

        # then the general case
        for i in range(2, N - 1):
            d[i] = (x[i + 1] - x[i - 2]) * x[i - 1] - x[i]

        # add the forcing term
        d = d + F

        # return the state derivatives
        return d

    def solveLorenz96RK4(self,timeSteps):
        r=ode(self.Lorenz96RK4).set_integrator("dopri5")
        r.set_initial_value(self.lorenzInitial,0)
        r.integrate(timeSteps)
        return r.y

    def solveLorenz96Int(self,timeSteps):
        r=odeint(self.Lorenz96Int,self.lorenzInitial,timeSteps)
        return r

def main():
    x=np.mat(np.identity(5))
    print(x[:,4])

    # l = Lorenz96Model()
    # y=l.lorenzInitial
    # print(y)
    # timeSteps=np.arange(0, 1000, 1)
    # # y=l.solveLorenz96RK4(100)
    # # print(y)
    # # w=l.solveLorenz96RK4(200)
    # # print(w-y)
    #
    #
    # x= l.solveLorenz96Int(timeSteps)
    # # print(z-y)
    # #
    # # plot first three variables
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot(x[900:1000, 0], x[900:1000, 1], x[900:1000, 2])
    # ax.set_xlabel('$x_1$')
    # ax.set_ylabel('$x_2$')
    # ax.set_zlabel('$x_3$')
    # plt.show()

    # from scipy.integrate import odeint
    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # # these are our constants
    # N = 36  # number of variables
    # F = 8  # forcing
    #
    # def Lorenz96(x, t):
    #     # compute state derivatives
    #     d = np.zeros(N)
    #     # first the 3 edge cases: i=1,2,N
    #     d[0] = (x[1] - x[N - 2]) * x[N - 1] - x[0]
    #     d[1] = (x[2] - x[N - 1]) * x[0] - x[1]
    #     d[N - 1] = (x[0] - x[N - 3]) * x[N - 2] - x[N - 1]
    #     # then the general case
    #     for i in range(2, N - 1):
    #         d[i] = (x[i + 1] - x[i - 2]) * x[i - 1] - x[i]
    #     # add the forcing term
    #     d = d + F
    #
    #     # return the state derivatives
    #     return d
    #
    # x0 = F * np.ones(N)  # initial state (equilibrium)
    # x0[19] += 0.01  # add small perturbation to 20th variable
    # t = np.arange(0,100, 1)
    #
    # x = odeint(Lorenz96, x0, t)
    # print(x[99,:])
    #
    # z = odeint(Lorenz96, x0, 100)
    # print(z)
    #
    # # # plot first three variables
    # # from mpl_toolkits.mplot3d import Axes3D
    # # fig = plt.figure()
    # # ax = fig.gca(projection='3d')
    # # ax.plot(x[:, 0], x[:, 1], x[:, 2])
    # # ax.set_xlabel('$x_1$')
    # # ax.set_ylabel('$x_2$')
    # # ax.set_zlabel('$x_3$')
    # # plt.show()

if __name__ == "__main__":
    main()
