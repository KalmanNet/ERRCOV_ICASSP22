
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

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
    x = np.array(10 * np.log10(1/np.square(q)))
    cov_EKF = np.log10(cov_EKF.detach().numpy()/10)
    MSE_EKF = np.log10(MSE_EKF.detach().numpy()/10)
    cov_knet = np.log10(cov_knet/10)
    MSE_knet = np.log10(MSE_knet/10)

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
    if x_true.shape[1] == 1:
        err = (x_true - x_est) ** 2
    else:
        err = torch.norm(x_true - x_est, dim=1) ** 2
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

def plot_error_evolution_trace(trace_EKF, trace_knet, emp_error_EKF, emp_error_knet, T, path, mismatch=False):
    
    t = np.arange(1, T+1)

    emp_error_EKF = torch.squeeze(emp_error_EKF).numpy()
    emp_error_knet = torch.squeeze(emp_error_knet).detach().numpy()

    trace_EKF = trace_EKF.numpy()
    trace_knet = trace_knet.numpy()

    short = True
    if short:
        t=np.arange(50, T+1)
        emp_error_EKF = emp_error_EKF[49:,]
        emp_error_knet = emp_error_knet[49:,]
        trace_EKF = trace_EKF[49:,]
        trace_knet = trace_knet[49:,]


    db = lambda x: 10 * np.log10(np.absolute(x))

    plt.close()
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8,5))

    ax[0].plot(t, db(trace_knet), '--r', label="KalmanNet Covariance Trace")
    ax[0].plot(t, db(trace_EKF), '--b', label="KF Covariance Trace")
    ax[0].plot(t, db(emp_error_knet), '-r', label="KalmanNet Empirical Error")
    ax[0].plot(t, db(emp_error_EKF), '-b', label="KF Empirical Error")


    ax[1].plot(t, db(emp_error_knet - trace_knet), '-r', label="KalmanNet Error Deviation")
    ax[1].plot(t, db(emp_error_EKF - trace_EKF), '-b', label="KF Error Deviation")

    plt.xlabel('t')

    ax[0].set_ylabel("Error [dB]")
    ax[1].set_ylabel("Error Deviation [dB]")

    ax[0].legend()
    ax[1].legend()

    if mismatch:
        plt.savefig(path + "error_evolution_mismatch.png", dpi=300)
    elif short:
        plt.savefig(path + "error_evolution_short.png", dpi=300)
    else:
        plt.savefig(path + "error_evolution.png", dpi=300)
    plt.show()


def plot_error_evolution(P_array_EKF, P_array_knet, emp_error_EKF, emp_error_knet, T, path, mismatch):
    
    t = np.arange(1, T+1)

    cov_EKF = torch.sqrt(torch.squeeze(torch.mean(P_array_EKF, dim=0))).detach().numpy()
    cov_knet = torch.sqrt(torch.squeeze(torch.mean(P_array_knet, dim=0))).detach().numpy()

    emp_error_EKF = torch.squeeze(emp_error_EKF).numpy()
    emp_error_knet = torch.squeeze(emp_error_knet).detach().numpy()

    db = lambda x: 10 * np.log10(np.absolute(x))


    short = True
    if short:
        t=np.arange(5, T+1)
        emp_error_EKF = emp_error_EKF[4:,]
        emp_error_knet = emp_error_knet[4:,]
        cov_EKF = cov_EKF[4:,]
        cov_knet = cov_knet[4:,]



    plt.close()
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8,5))

    ax[0].plot(t, db(cov_knet), '--r', label="KalmanNet Prediced Error")
    ax[0].plot(t, db(cov_EKF), '--b', label="KF Theoretical Error")
    ax[0].plot(t, db(emp_error_knet), '-r', label="KalmanNet Empirical Error")
    ax[0].plot(t, db(emp_error_EKF), '-b', label="KF Empirical Error")


    ax[1].plot(t, db(emp_error_knet - cov_knet), '-r', label="KalmanNet Error Deviation")
    ax[1].plot(t, db(emp_error_EKF - cov_EKF), '-b', label="KF Error Deviation")

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
