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

sys.path.insert(1, 'src/Results/')
from MSE import MSE

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

    model_mismatch = True
    train = False
    print_idx = 1
    only_one_GT = False


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



    # Generate Data
    load = True
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
                [test_input_mult, test_target_mult] = Data_Gen_multiple_obs(sys_model, 200, 5)
                torch.save({"All Data": [test_input_mult, test_target_mult]}, data_gen_file_mult)
                print("Finish Data Gen")
        else:
            print("Start Data Gen")
            [test_input_mult, test_target_mult] = Data_Gen_multiple_obs(sys_model, 200, 5)
            torch.save({"All Data": [test_input_mult, test_target_mult]}, data_gen_file_mult)
            print("Finish Data Gen")


    # EKF results
    if only_one_GT:
        [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, KG_array, EKF_out, P_array_EKF, _] = EKFTest(sys_model_inacc, test_input, test_target, modelKnowledge = 'full', calculate_covariance=True)
        print(f"EKF: MSE={MSE_EKF_dB_avg} [dB]")


    ## Build Neural Network
    Model = NNBuild(sys_model_inacc)

    ## Train Neural Network
    if train:
        [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = NNTrain(sys_model_inacc, Model, cv_input, cv_target, train_input, train_target, path_results)
    
    ## Test Neural Network
    if only_one_GT:
        [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, KGain_array, x_out_array, P_array, _] = NNTest(sys_model_inacc, test_input, test_target, path_results, calculate_covariance=True)

        emp_err_EKF = empirical_error(test_input, EKF_out)
        emp_err_knet = empirical_error(test_input, x_out_array)

        print(f"EKF: MSE={MSE_EKF_dB_avg} [dB]")
        print(f"EKF: MSE={MSE_EKF_linear_avg}")
        print(f"KalmanNet: MSE={MSE_test_dB_avg} [dB]")
        print(f"KalmanNet: MSE={MSE_test_linear_avg}")

        avg_cov_EKF = torch.mean(P_array_EKF, (0,1))
        avg_cov_knet = torch.mean(P_array, (0,1)).detach()
        #print(f"EKF state covariance: {torch.squeeze(P_array_EKF[print_idx,]).detach().numpy()}")
        #print(f"KalmanNet state covariance: {torch.squeeze(P_array[print_idx,]).detach().numpy()}")
        print(f"EKF average covariance: {avg_cov_EKF}")
        print(f"KalmanNet average covariance: {avg_cov_knet}")
        print(f"trace EKF: {torch.trace(avg_cov_EKF)}")
        print(f"trace KalmanNet: {torch.trace(avg_cov_knet)}")
        
        plot_confidence(test_target, P_array_EKF, P_array, emp_err_EKF, emp_err_knet, T, path_results, model_mismatch)

        """
        if n == 1:
            #plot_trajectory1d(print_idx, test_target, x_out_array, P_array, T, "KalmanNet", "KalmanNet", path_results)
            #plot_trajectory1d(print_idx, test_target, EKF_out, P_array_EKF, T, "EKF", "EKF", path_results)
            plot_all_trajectories(test_target, EKF_out, x_out_array, P_array_EKF, P_array, T, path_results)
        elif n == 2:
            plot_trajectory2d(print_idx, test_target, x_out_array, P_array, T, "KalmanNet", "KalmanNet", path_results, n=100)
            plot_trajectory2d(print_idx, test_target, EKF_out, P_array_EKF, T, "EKF", "EKF", path_results, n=100)

        plot_error(P_array_EKF, P_array, emp_err_EKF, emp_err_knet, T, path_results)
        plot_error_difference(P_array_EKF, P_array, emp_err_EKF, emp_err_knet, T, path_results)
        """


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
                #r.write(f"EKF covariance error: {cov_err_EKF/N}\n\n")
                r.write(f"MSE KalmanNet: {10**(MSE_test_dB_avg/10)}, ({MSE_test_dB_avg} dB)\n")
                r.write(f"KalmanNet average covariance: {avg_cov_knet.numpy()}\n")
                #r.write(f"KalmanNet covariance error: {cov_err_KNet/N}\n")

        """
        q_vals = [10., 5., 2., 1., .5, .2, .1]
        num_q = len(q_vals)
        MSE_EKF = torch.zeros((num_q))
        cov_EKF = torch.zeros((num_q))
        for i, q in enumerate(q_vals):
            Q_tmp = (q ** 2) * torch.eye(n)
            sys_model_EKF = SystemModel(f, Q_tmp, h, R, T)
            sys_model_EKF.InitSequence(m1x_0, m2x_0)
            [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, KG_array, EKF_out, P_array_EKF, _] = EKFTest(sys_model_EKF, test_input, test_target, modelKnowledge = 'full', calculate_covariance=True)
            MSE_EKF[i] = MSE_EKF_linear_avg
            cov_EKF[i] = P_array_EKF[0, -1, 0, 0]
        print(MSE_EKF, cov_EKF)
        plot_q_mismatch(q_vals, MSE_EKF, cov_EKF, MSE_test_linear_avg, avg_cov_knet, path_results)

    """


    else:
        [_, _, _, _, EKF_out_mult, P_array_EKF_mult, _] = EKFTest(sys_model_inacc, test_input_mult, test_target_mult, modelKnowledge = 'full', calculate_covariance=True)
        [_, _, _, _, x_out_array_mult, P_array_mult, _] = NNTest(sys_model_inacc, test_input_mult, test_target_mult, path_results, calculate_covariance=True)

        emp_err_EKF_mult = empirical_error(test_input_mult, EKF_out_mult)
        emp_err_knet_mult = empirical_error(test_input_mult, x_out_array_mult)
        plot_error_evolution(P_array_EKF_mult, P_array_mult, emp_err_EKF_mult, emp_err_knet_mult, T, path_results, model_mismatch)


def plot_q_mismatch(q, MSE_EKF, cov_EKF, MSE_knet, cov_knet, path):

    """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
    })
    """
    x = np.array(10 * np.log(1/np.square(q)))
    cov_EKF = np.log(cov_EKF.detach().numpy()/10)
    MSE_EKF = np.log(MSE_EKF.detach().numpy()/10)
    cov_knet = np.log(cov_knet/10)
    MSE_knet = np.log(MSE_knet/10)

    plt.close()
    plt.plot(
            x, MSE_EKF, '-rs',
            x, cov_EKF, '-ms',
            [1], MSE_knet, 'b*',
            [1], cov_knet, 'c*')
    plt.xlabel(r'$\displaystyle\frac{1}{q^2} [dB]$')
    plt.tight_layout()
    plt.autoscale()
    plt.gcf().subplots_adjust(left=0.1)
    plt.ylabel('Error [dB]')
    plt.legend(['KF MSE', 'KF Theoretical Error', 'KalmanNet MSE', 'KalmanNet Predicted Error'])
    plt.grid()
    plt.savefig(path + 'error_vs_q.png', dpi=300)
    plt.show()


def empirical_error(x_true, x_est):
    err = (x_true - x_est) ** 2
    err = torch.mean(err, dim=0)
    return torch.sqrt(err)


def plot_confidence(x_true, cov_EKF, cov_knet, err_EKF, err_knet, T, path, mismatch):

    t = np.arange(1, T+1)
    x = torch.squeeze(x_true[0,]).numpy()

    err_EKF = torch.squeeze(err_EKF).numpy()
    err_knet = torch.squeeze(err_knet).detach().numpy()

    cov_EKF = torch.sqrt(torch.squeeze(torch.mean(cov_EKF, dim=0))).numpy()
    cov_knet = torch.sqrt(torch.squeeze(torch.mean(cov_knet, dim=0))).detach().numpy()

    plt.close()
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8,5))

    ax[0].plot(t, x, '-k', label="Ground Truth")

    ax[0].plot(t, x + err_EKF, '-r')
    ax[0].plot(t, x - err_EKF, '-r', label="KF Empirical Error")

    ax[0].plot(t, x + cov_EKF, '--b')
    ax[0].plot(t, x - cov_EKF, '--b', label="KF Theoretical Error")


    ax[1].plot(t, x, '-k', label="Ground Truth")

    ax[1].plot(t, x + err_knet, '-r')
    ax[1].plot(t, x - err_knet, '-r', label="KalmanNet Empirical Error")

    ax[1].plot(t, x + cov_knet, '--b')
    ax[1].plot(t, x - cov_knet, '--b', label="KalmanNet Predicted Error")


    plt.xlabel('t')

    ax[0].set_ylabel('x')
    ax[1].set_ylabel('x')

    ax[0].legend()
    ax[1].legend()

    if mismatch:
        plt.savefig(path + "confidence_mismatch.png", dpi=300)
    else:
        plt.savefig(path + "confidence.png", dpi=300)
    plt.show()


def plot_error_evolution(P_array_EKF, P_array_knet, emp_error_EKF, emp_error_knet, T, path, mismatch):
    
    t = np.arange(1, T+1)

    cov_EKF = torch.sqrt(torch.squeeze(torch.mean(P_array_EKF, dim=0))).detach().numpy()
    cov_knet = torch.sqrt(torch.squeeze(torch.mean(P_array_knet, dim=0))).detach().numpy()

    emp_error_EKF = torch.squeeze(emp_error_EKF).numpy()
    emp_error_knet = torch.squeeze(emp_error_knet).detach().numpy()

    db = lambda x: 10 * np.log(x)

    cov_EKF = db(cov_EKF)
    cov_knet = db(cov_knet)
    emp_error_EKF = db(emp_error_EKF)
    emp_error_knet = db(emp_error_knet)

    plt.close()
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8,5))

    ax[0].plot(t, cov_knet, '-r', label="KalmanNet Prediced Error")
    ax[0].plot(t, cov_EKF, '-b', label="KF Theoretical Error")


    ax[1].plot(t, emp_error_knet - cov_knet, '-r', label="KalmanNet Error Deviation")
    ax[1].plot(t, emp_error_EKF - cov_EKF, '-b', label="KF Error Deviation")

    plt.xlabel('t')

    ax[0].set_ylabel("Error [dB]")
    ax[1].set_ylabel("Error Deviation [dB]")

    ax[0].legend()
    ax[1].legend()

    if mismatch:
        plt.savefig(path + "error_evolution_mismatch.png", dpi=300)
    else:
        plt.savefig(path + "error_evolution.png", dpi=300)
    plt.show()




def plot_trajectory1d(idx, x_true, x, P_array, T, label, title, path):

    t = np.arange(1, T+1)
    x_GT = torch.squeeze(x_true[idx,]).numpy()
    x = torch.squeeze(x[idx,]).detach().numpy()
    err = torch.squeeze(P_array[idx,]).detach().numpy()
    err_upper = x + np.sqrt(err)
    err_lower = x - np.sqrt(err)

    plt.close()
    plt.plot(t, x_GT, '-r', t, x, '-b')
    plt.fill_between(t, err_upper, err_lower, alpha=0.2)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title(title)
    plt.legend(['Ground truth', label])
    plt.savefig(path + f"{label}.png", dpi=300)
    plt.show()

def plot_all_trajectories(x_true, x_EKF, x_knet, cov_EKF, cov_knet, T, path):

    t = np.arange(1, T+1)
    x_GT = torch.squeeze(x_true[0,]).numpy()

    x_EKF = torch.squeeze(x_EKF[:2,]).detach().numpy()
    err_EKF = torch.squeeze(cov_EKF[:2,]).detach().numpy()
    upper_EKF = x_EKF + np.sqrt(err_EKF)
    lower_EKF = x_EKF - np.sqrt(err_EKF)

    x_knet = torch.squeeze(x_knet[:2,]).detach().numpy()
    err_knet = torch.squeeze(cov_knet[:2,]).detach().numpy()
    upper_knet = x_knet + np.sqrt(err_knet)
    lower_knet = x_knet - np.sqrt(err_knet)

    plt.close()
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8,5))
    ax[0].plot(t, x_GT, '-k')

    ax[0].plot(t, x_EKF[0,], '-r')
    ax[0].fill_between(t, upper_EKF[0,], lower_EKF[0,], alpha=0.2, color='r')

    ax[0].plot(t, x_EKF[1,], '-m')
    ax[0].fill_between(t, upper_EKF[1,], lower_EKF[1,], alpha=0.2, color='m')

    plt.xlabel('t')
    ax[0].set_ylabel('x')
    ax[0].set_title('State Estimates with Error: KF')
    #ax[0].legend(['Ground Truth', 'KF', 'KF predicted confidence'])
    ax[0].legend(['Ground Truth', 'KF (1)', 'KF predicted confidence (1)',  'KF (2)', 'KF predicted confidence (2)'])

    ax[1].plot(t, x_GT, '-k')

    ax[1].plot(t, x_knet[0,], '-b')
    ax[1].fill_between(t, upper_knet[0,], lower_knet[0,], alpha=0.2, color='b')

    ax[1].plot(t, x_knet[1,], '-c')
    ax[1].fill_between(t, upper_knet[1,], lower_knet[1,], alpha=0.2, color='c')

    ax[1].set_ylabel('x')
    ax[1].set_title('State Estimates with Error: KalmanNet')
    #ax[1].legend(['Ground Truth', 'KalmanNet', 'KalmanNet predicted confidence'])
    ax[1].legend(['Ground Truth', 'KalmanNet (1)', 'KalmanNet predicted confidence (1)',  'KalmanNet (2)', 'KalmanNet predicted confidence (2)'])
    plt.savefig(path + "trajectories.png", dpi=300)
    plt.show()




def plot_trajectory2d(idx, x_true, x, P_array, T, label, title, path, n=20):

    x_GT = torch.squeeze(x_true[idx,:,:n]).numpy()
    x = torch.squeeze(x[idx,:,:n]).detach().numpy()

    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_GT[0,], x_GT[1,], '+r', x[0,], x[1,], '+b')
    for i in range(n):
        plot_ellipse(x[0,i], x[1,i], P_array[idx, i, :, :].detach(), ax)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.legend(['Ground truth', label])
    plt.savefig(path + f"{label}.png", dpi=300)
    plt.show()

def plot_ellipse(x, y, P, ax):

    vals, vecs = np.linalg.eig(P.numpy())
    if vals[0] < vals[1]:
        vals = vals[::-1]
        vecs = vecs[:,::-1]

    major = 2 * np.sqrt(vals[0]*5.991)
    minor = 2 * np.sqrt(vals[1]*5.991)

    angle = np.arctan(vecs[1,0]/vecs[0,0])
    transf = transforms.Affine2D().rotate(angle).translate(x,y) + ax.transData
    ellipse = Ellipse((0, 0), major, minor, alpha=0.2)
    ellipse.set_transform(transf)
    ax.add_patch(ellipse)

def plot_error(P_array_EKF, P_array_knet, emp_error_EKF, emp_error_knet, T, path):

    t = np.arange(1, T+1)

    cov_EKF = torch.sqrt(torch.squeeze(torch.mean(P_array_EKF, dim=0))).detach().numpy()
    cov_knet = torch.sqrt(torch.squeeze(torch.mean(P_array_knet, dim=0))).detach().numpy()

    emp_error_EKF = torch.squeeze(emp_error_EKF).detach().numpy()
    emp_error_knet = torch.squeeze(emp_error_knet).detach().numpy()

    plt.close()
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8,5))
    ax[0].plot(t, cov_EKF, '-r', t, emp_error_EKF, '-m')
    ax[0].legend(['KF error estimate', 'KF empirical error'])
    ax[0].set_ylabel('covariance')

    ax[1].plot(t, cov_knet, '-b', t, emp_error_knet, '-c')
    ax[1].legend(['KalmanNet error estimate', 'KalmanNet empirical error'])
    ax[1].set_ylabel('covariance')

    plt.xlabel('t')

    plt.title('error vs. time')
    plt.savefig(path + "error_evolution.png", dpi=300)
    plt.show()

def plot_error_difference(P_array_EKF, P_array_knet, emp_error_EKF, emp_error_knet, T, path):

    t = np.arange(1, T+1)

    cov_EKF = torch.sqrt(torch.squeeze(torch.mean(P_array_EKF, dim=0))).detach().numpy()
    cov_knet = torch.sqrt(torch.squeeze(torch.mean(P_array_knet, dim=0))).detach().numpy()

    emp_error_EKF = torch.squeeze(emp_error_EKF).detach().numpy()
    emp_error_knet = torch.squeeze(emp_error_knet).detach().numpy()

    err_diff_EKF = emp_error_EKF  - cov_EKF 
    err_diff_knet = emp_error_knet - cov_knet 

    plt.close()
    plt.plot(t, err_diff_EKF, '-r', t, err_diff_knet, '-b')
    plt.xlabel('t')
    plt.ylabel('deviation from predicted error')
    plt.title('empirical error minus predicted error')
    plt.legend(['KF', 'KalmanNet'])

    plt.savefig(path + "error_deviation.png", dpi=300)
    plt.show()

if __name__== "__main__":
    main()
