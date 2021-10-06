import KalmanNet_sysmdl

from KalmanNet_data import DataGen, DecimateData

from EKF_test import EKFTest

from KalmanNet_build import NNBuild
from KalmanNet_train import NNTrain
from KalmanNet_test import NNTest
from KalmanNet_plt import NNPlot_train, plotTrajectories

from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from DataGenerator import load_data

from filing_paths import path_model, path_session
from NN_parameters import N_T, N_E, N_Epochs, N_CV, N_B, learning_rate, wd

from KalmanNet_nn import in_mult, out_mult

from scipy.io import savemat

import os

from plotCovariance import plot_confidence, plot_error_evolution, empirical_error, plot_error_evolution_trace

import sys
sys.path.insert(1, path_model)
from parameters import Q_design, R_design, Q_mod, R_mod, T_test, J, J_mod, delta_t_gen
from parameters import m1x_0_design, m2x_0_design, m1x_0_mod, m2x_0_mod, T, lambda_q, lambda_r
from parameters import m1x_0_design_test, m1x_0_mod_test, T_mod, T_test_mod, roll_deg, delta_t_mod, m, n, delta_t
from model import f_interpolate, h, f

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    dev = torch.device("cpu")


def main():

    T=100

    # Set to True to retrain KalmanNet
    train_KNet = False

    # define paths to save results
    path_base = path_session
    os.makedirs(path_base, exist_ok=True)
    results_file = path_base + "results.pt"
    path_results = path_base
    

    # load 
    [test_target, test_input, test_IC, _, _, _, _, _, _, train_target, train_input, train_IC, CV_target, CV_input, CV_IC] = load_data("identity", 0, process_noise=None)

    R = torch.eye(n)

    # Dynamical model
    sys_model = KalmanNet_sysmdl.SystemModel(f_interpolate, Q_mod, h, R_mod, T)
    sys_model.InitSequence(m1x_0_design, m2x_0_design)
    sys_model_EKF = KalmanNet_sysmdl.SystemModel(f_interpolate, Q_mod, h, R, T)
    sys_model_EKF.InitSequence(m1x_0_design, m2x_0_design)

    # testing EKF
    [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, KG_array, EKF_out, P_array_EKF, _]= EKFTest(sys_model_EKF, test_input, test_target, init_cond=test_IC, calculate_covariance=True)

    if train_KNet:
        # Build and train KalmanNet
        Model = NNBuild(sys_model)
        [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = NNTrain(sys_model, Model, CV_input, CV_target, train_input, train_target, path_base, sequential_training=True, train_IC=train_IC, CV_IC=CV_IC)

    # Evaluate KalmanNet
    [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, KGain_array, x_out_array, P_array, _] = NNTest(sys_model, test_input, test_target, path_base, IC=test_IC, calculate_covariance=True)


    # Print results and generate plots
    emp_err_EKF = empirical_error(test_target, EKF_out)
    emp_err_knet = empirical_error(test_target, x_out_array)

    print(f"EKF: MSE={MSE_EKF_dB_avg} [dB]")
    print(f"KalmanNet: MSE={MSE_test_dB_avg} [dB]")

    avg_cov_EKF = torch.mean(P_array_EKF, 0)
    avg_cov_knet = torch.mean(P_array, 0).detach()

    trace_EKF = calc_trace(avg_cov_EKF)
    trace_knet = calc_trace(avg_cov_knet)

    plot_error_evolution_trace(trace_EKF, trace_knet, emp_err_EKF, emp_err_knet, T, path_results)

    MSE = torch.norm(test_target - x_out_array, dim=1) ** 2
    MSE = torch.flatten(MSE).detach().numpy()

    plt.close()
    plt.hist(MSE, bins=100, density=True)
    plt.xlabel("MSE")
    plt.ylabel("probability")
    plt.savefig(path_results + 'histogram.png', dpi=300)
    plt.show()



def calc_trace(A):
    l = A.shape[0]
    out = torch.empty(l)
    for k in range(l):
        out[k] = torch.trace(A[k,])
    return out


if __name__== "__main__":
    main()
