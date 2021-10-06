from KalmanNet_data import DataGen, Data_Gen_multiple_obs
from KalmanNet_sysmdl import SystemModel
from EKF_test import EKFTest

from KalmanNet_build import NNBuild
from KalmanNet_train import NNTrain
from KalmanNet_test import NNTest
from KalmanNet_plt import NNPlot_train, plotTrajectories

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from filing_paths import path_model, path_session
from NN_parameters import learning_rate, N_CV, N_Epochs, N_T, N_E, N_B, wd

import sys
sys.path.insert(1, path_model)
from model import f, h, fInacc, hInacc
from parameters import m, n, m1x_0, m2x_0, T, Q, R, Q_mod, R_mod, sigma_r, sigma_q, F, H, F_mod, H_mod

from plotCovariance import plot_confidence, plot_error_evolution, empirical_error 

sys.path.insert(1, 'src/Results/')
#from MSE import MSE

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

def main():

    ############################
    ### parameters to be set ###
    ############################

    # If True, the model used by the KF and KalmanNet differs from the one used to generate the data to simulate a mismatched model
    model_mismatch = False

    # If True, every test trajectory is a different noisy observation from the same ground truth trajectory.
    # This setting is used to generate the confidence interval plot.
    # If False, every test trajectory uses a unique ground truth trajectory.
    # This setting is used to generate the error evolution plot.
    only_one_GT = True
    
    # Set to True to retrain KalmanNet, otherwise the stored model is loaded
    # only use train = True in combination with only_one_GT = True
    train = False

    if train:
        assert only_one_GT, "to train, set only_one_GT to True"


    # defining file paths
    path = path_session + f"r_{str(sigma_r)}_q_{str(sigma_q)}/"
    if model_mismatch:
        path_results = path + 'model_mismatch/Results/'
    else:
        path_results = path + 'full_knowledge/Results/'
    data_gen_file = path + "data_gen.pt"
    data_gen_file_mult = path + "data_gen_mult.pt"
    results_file = path_results + "results.pt"
    results_file_mult = path_results + "results_mult.pt"


    # Dynamic model
    sys_model = SystemModel(f, Q, h, R, T)
    sys_model.InitSequence(m1x_0, m2x_0)

    # inaccurate dynamic model
    if model_mismatch:
        sys_model_inacc = SystemModel(fInacc, Q_mod, hInacc, R_mod, T)
        sys_model_inacc.InitSequence(m1x_0, m2x_0)
    else:
        sys_model_inacc = sys_model



    # Generate or load data
    load = True # set to False to generate new dataset
    if only_one_GT:
        if load:
            try:
                print("Start Loading Data")
                data_file = torch.load(data_gen_file)
                [train_input, train_target, cv_input, cv_target,test_input, test_target] = data_file["All Data"]
                print("Finish Loading Data")
            except:
                print("Start Data Gen")
                [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataGen(sys_model, repeat_test_trajectory=True)
                torch.save({"All Data":[train_input, train_target, cv_input, cv_target, test_input, test_target]}, data_gen_file)

        else:
            print("Start Data Gen")
            [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataGen(sys_model, repeat_test_trajectory=True)
            torch.save({"All Data":[train_input, train_target, cv_input, cv_target, test_input, test_target]}, data_gen_file)

    else:
        if load:
            try:
                print("Start Loading Data")
                data_file = torch.load(data_gen_file_mult)
                [test_input_mult, test_target_mult] = data_file["All Data"]
                print("Finish Loading Data")
            except:
                print("Start Data Gen")
                #[test_input_mult, test_target_mult] = Data_Gen_multiple_obs(sys_model, 200, 5)
                [_, _, _, _, test_input_mult, test_target_mult] = DataGen(sys_model, repeat_test_trajectory=False)
                torch.save({"All Data": [test_input_mult, test_target_mult]}, data_gen_file_mult)
                print("Finish Data Gen")
        else:
            print("Start Data Gen")
            #[test_input_mult, test_target_mult] = Data_Gen_multiple_obs(sys_model, 200, 5)
            [_, _, _, _, test_input_mult, test_target_mult] = DataGen(sys_model, repeat_test_trajectory=False)
            torch.save({"All Data": [test_input_mult, test_target_mult]}, data_gen_file_mult)
            print("Finish Data Gen")

    # Build Neural Network
    Model = NNBuild(sys_model_inacc)

    # Train Neural Network
    if train:
        [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = NNTrain(sys_model_inacc, Model, cv_input, cv_target, train_input, train_target, path_results)
    
    # Case with single ground truth trajectory with multiple observations
    if only_one_GT:
        # Test classical Kalman filter
        [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, KG_array, EKF_out, P_array_EKF, _] = EKFTest(sys_model_inacc, test_input, test_target, modelKnowledge = 'full', calculate_covariance=True)
        print(f"EKF: MSE={MSE_EKF_dB_avg} [dB]")

        # Test KalmanNet
        [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, KGain_array, x_out_array, P_array, _] = NNTest(sys_model_inacc, test_input, test_target, path_results, calculate_covariance=True)

        # print out results and generate plots
        emp_err_EKF = empirical_error(test_target, EKF_out)
        emp_err_knet = empirical_error(test_target, x_out_array)

        print(f"EKF: MSE={MSE_EKF_dB_avg} [dB]")
        print(f"KalmanNet: MSE={MSE_test_dB_avg} [dB]")

        avg_cov_EKF = torch.mean(P_array_EKF, (0,1))
        avg_cov_knet = torch.mean(P_array, (0,1)).detach()
        print(f"EKF average covariance: {avg_cov_EKF}")
        print(f"KalmanNet average covariance: {avg_cov_knet}")
        
        plot_confidence(test_target, P_array_EKF, P_array, emp_err_EKF, emp_err_knet, T, path_results, model_mismatch)

        # write results to file
        if train:
            EKF_results = [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, KG_array, EKF_out, P_array_EKF]
            KNet_training_results = [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch]
            KNet_testing_results = [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, KGain_array, x_out_array, P_array]
            torch.save({"EKF":EKF_results, "KNet_training":KNet_training_results, "KNet_testing": KNet_testing_results}, results_file)

            with open(path_results + "results.txt",'w') as r:
                r.write("Dynamic model\n")
                r.write("------------------------------------------\n")
                r.write("true model:\n")
                r.write(f"F = {F.numpy()}\n")
                r.write(f"H = {H.numpy()}\n")
                r.write(f"Q = {Q.numpy()}\n")
                r.write(f"R = {R.numpy()}\n")
                r.write("model used by KalmanNet and Kalman filter:\n")
                r.write(f"F_mod = {F_mod.numpy()}\n")
                r.write(f"H_mod = {H_mod.numpy()}\n")
                r.write(f"Q_mod = {Q_mod.numpy()}\n")
                r.write(f"R_mod = {R_mod.numpy()}\n\n")
                r.write("training parameters:\n")
                r.write("------------------------------------------\n")
                r.write(f"learning rate = {learning_rate}\n")
                r.write(f"weight decay = {wd}\n\n")
                r.write("evaluation results\n")
                r.write("------------------------------------------\n")
                r.write(f"MSE EKF: {10**(MSE_EKF_dB_avg/10)} ({MSE_EKF_dB_avg} dB)\n")
                r.write(f"EKF average covariance: {avg_cov_EKF.numpy()}\n")
                r.write(f"MSE KalmanNet: {10**(MSE_test_dB_avg/10)}, ({MSE_test_dB_avg} dB)\n")
                r.write(f"KalmanNet average covariance: {avg_cov_knet.numpy()}\n")



    # case with multiple unique ground truth trajectories
    else:
        # Test classical Kalman filter
        [MSE_EKF_linear_arr_mult, MSE_EKF_linear_avg_mult, MSE_EKF_dB_avg_mult, KG_array_mult, EKF_out_mult, P_array_EKF_mult, _] = EKFTest(sys_model_inacc, test_input_mult, test_target_mult, modelKnowledge = 'full', calculate_covariance=True)


        [MSE_test_linear_arr_mult, MSE_test_linear_avg_mult, MSE_test_dB_avg_mult, KGain_array_mult, x_out_array_mult, P_array_mult, _] = NNTest(sys_model_inacc, test_input_mult, test_target_mult, path_results, calculate_covariance=True)

        # print out results and generate plots
        emp_err_EKF_mult = empirical_error(test_target_mult, EKF_out_mult)
        emp_err_knet_mult = empirical_error(test_target_mult, x_out_array_mult)

        # Test KalmanNet
        MSE = (test_target_mult - x_out_array_mult) ** 2
        MSE = torch.flatten(MSE).detach().numpy()
        plt.close()
        plt.hist(MSE, bins=100, density=True)
        plt.xlabel("MSE")
        plt.ylabel("probability")
        if model_mismatch:
            plt.savefig(path_results + 'histogram_linear_mismatch.png', dpi=300)
        else:
            plt.savefig(path_results + 'histogram_linear.png', dpi=300)
        plt.show()

        plot_error_evolution(P_array_EKF_mult, P_array_mult, emp_err_EKF_mult, emp_err_knet_mult, T, path_results, model_mismatch)

        
if __name__== "__main__":
    main()
