import numpy as np
import torch.nn as nn
import torch
import time

from EKF import ExtendedKalmanFilter

from NN_parameters import N_E, N_CV, N_T

def EKFTest(SysModel, test_input, test_target, modelKnowledge = 'full', allStates=True, init_cond=None, calculate_covariance=False):

    N_T = test_target.size()[0]

    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_EKF_linear_arr = np.empty(N_T)

    EKF = ExtendedKalmanFilter(SysModel, modelKnowledge)
    EKF.InitSequence(SysModel.m1x_0, SysModel.m2x_0, calculate_covariance=calculate_covariance)

    KG_array = torch.zeros_like(EKF.KG_array)
    EKF_out = torch.empty([N_T, SysModel.m, SysModel.T])

    if calculate_covariance:
        P_array = torch.empty(N_T, SysModel.T, SysModel.n, SysModel.n)

    start = time.time()
    for j in range(0, N_T):
        if init_cond is not None:
            EKF.InitSequence(torch.unsqueeze(init_cond[j, :], 1), SysModel.m2x_0, calculate_covariance=calculate_covariance)
        EKF.GenerateSequence(test_input[j, :, :])

        if(allStates):
            MSE_EKF_linear_arr[j] = loss_fn(EKF.x, test_target[j, :, :]).item()
        else:
            loc = torch.tensor([True,False,True,False])
            MSE_EKF_linear_arr[j] = loss_fn(EKF.x[loc,:], test_target[j, :, :]).item()
        KG_array = torch.add(EKF.KG_array, KG_array) 
        EKF_out[j,:,:] = EKF.x

        if calculate_covariance:
            P_array[j, :, :, :] = EKF.P_array

    end = time.time()
    t = end - start

    # Average KG_array over Test Examples
    KG_array /= N_T

    MSE_EKF_linear_avg = np.mean(MSE_EKF_linear_arr)
    MSE_EKF_dB_avg = 10 * np.log10(MSE_EKF_linear_avg)

    if calculate_covariance:
        return [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, KG_array, EKF_out, P_array, t]
    else:
        return [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, KG_array, EKF_out, t]



