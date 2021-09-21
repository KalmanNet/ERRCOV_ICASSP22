"""# **Class: Extended Kalman Filter**
Theoretical Non Linear Kalman
"""
import torch
import numpy as np

from filing_paths import path_model

import sys
sys.path.insert(1, path_model)
from model import getJacobian

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    dev = torch.device("cpu")


class ExtendedKalmanFilter:

    def __init__(self, SystemModel, mode='full'):
        self.f = SystemModel.f
        self.m = SystemModel.m

        # Has to be transformed because of EKF non-linearity
        self.Q = SystemModel.Q

        self.h = SystemModel.h
        self.n = SystemModel.n

        # Has to be transofrmed because of EKF non-linearity
        self.R = SystemModel.R

        self.T = SystemModel.T

        self.calculate_covariance = False

        # Pre allocate an array for predicted state
        self.x = torch.empty(size=[self.m, self.T]).to(dev, non_blocking=True)

        # Pre allocate KG array
        self.KG_array = torch.zeros((self.T,self.m,self.n)).to(dev, non_blocking=True)
        self.P_array = torch.empty((self.T, self.n, self.n))
        self.i = 0 # Index for KG_array alocation

    # Predict
    def Predict(self):
        # Predict the 1-st moment of x
        self.m1x_prior = self.f(self.m1x_posterior)
        # Compute the Jacobians
        #self.UpdateJacobians(getJacobian(self.m1x_posterior,self.f), getJacobian(self.m1x_prior, self.h))
        self.UpdateJacobians(getJacobian(self.m1x_posterior,self.f), getJacobian(self.m1x_prior, self.h))
        # Predict the 2-nd moment of x
        self.m2x_prior = torch.matmul(self.F, self.m2x_posterior)
        self.m2x_prior = torch.matmul(self.m2x_prior, self.F_T) + self.Q

        # Predict the 1-st moment of y
        self.m1y = self.h(self.m1x_prior)
        # Predict the 2-nd moment of y
        self.m2y = torch.matmul(self.H, self.m2x_prior)
        self.m2y = torch.matmul(self.m2y, self.H_T) + self.R

    # Compute the Kalman Gain
    def KGain(self):
        self.KG = torch.matmul(self.m2x_prior, self.H_T)
        self.KG = torch.matmul(self.KG, torch.inverse(self.m2y))

        #Save KalmanGain
        self.KG_array[self.i] = self.KG
        self.i += 1

    # Innovation
    def Innovation(self, y):
        y = y.to(dev, non_blocking=True)
        self.dy = y - self.m1y

    # Compute Posterior
    def Correct(self):
        # Compute the 1-st posterior moment
        self.m1x_posterior = self.m1x_prior + torch.matmul(self.KG, self.dy)

        # Compute the 2-nd posterior moment
        self.m2x_posterior = torch.matmul(self.m2y, torch.transpose(self.KG, 0, 1))
        self.m2x_posterior = self.m2x_prior - torch.matmul(self.KG, self.m2x_posterior)

        if self.calculate_covariance:
            self.P_array[self.i-1,:,:] = self.m2x_posterior

    def Update(self, y):
        self.Predict()
        self.KGain()
        self.Innovation(y)
        self.Correct()

        return self.m1x_posterior

    def InitSequence(self, m1x_0, m2x_0, calculate_covariance=False):
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0

        self.calculate_covariance = calculate_covariance


    def UpdateJacobians(self, F, H):
        self.F = F
        self.F_T = torch.transpose(F,0,1)
        self.H = H
        self.H_T = torch.transpose(H,0,1)
        #print(self.H,self.F,'\n')

    #########################
    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, y):
        self.m1x_posterior = self.m1x_0
        self.m2x_posterior = self.m2x_0

        self.i = 0

        for t in range(0, self.T):
            yt = torch.unsqueeze(y[:, t], 1)
            xt = self.Update(yt)
            self.x[:, t] = torch.squeeze(xt)
