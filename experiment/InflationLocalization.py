import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
from diffEqModel.Lorenz96Model import Lorenz96Model
import math

class InflationLocalization:
    def __init__(self,nEnsembles=20,lorenzDim=40):
        self.nEnsembles=nEnsembles
