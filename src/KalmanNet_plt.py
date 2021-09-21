import numpy as np
import matplotlib
#matplotlib.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch

from NN_parameters import N_Epochs

N_Epochs_plt = N_Epochs

# Legend
Klegend = ["KNet - Train", "KNet - CV", "KNet - Test", "Extended Kalman Filter", "Baseline"]
Klegend_partial = ["KNet - Train", "KNet - CV", "KNet - Test", "Extended Kalman Filter Full", "Baseline", "Extended Kalman Filter Partial"]
# Color
KColor = ['ro', 'yo', 'g-', 'b-', 'y-']
KColor_partial = ['ro', 'yo', 'g-', 'b-', 'y-', 'r-']

def NNPlot_train(MSE_KF_linear_arr, MSE_KF_dB_avg,
                 MSE_test_linear_arr, MSE_test_dB_avg,
                 MSE_cv_dB_epoch, MSE_train_dB_epoch, 
                 MSE_baseline_dB_avg, file_path):

    N_Epochs_plt = np.shape(MSE_cv_dB_epoch)[0]

    ###########################
    ### Plot per epoch [dB] ###
    ###########################
    plt.figure(figsize = (50, 20))

    x_plt = range(0, N_Epochs_plt)

    # Train
    y_plt1 = MSE_train_dB_epoch[range(0, N_Epochs_plt)]
    plt.plot(x_plt, y_plt1, KColor[0], label=Klegend[0])

    # CV
    y_plt2 = MSE_cv_dB_epoch[range(0, N_Epochs_plt)]
    plt.plot(x_plt, y_plt2, KColor[1], label=Klegend[1])

    # KNet - Test
    y_plt3 = MSE_test_dB_avg * np.ones(N_Epochs_plt)
    plt.plot(x_plt, y_plt3, KColor[2], label=Klegend[2])

    # EKF
    y_plt4 = MSE_KF_dB_avg * np.ones(N_Epochs_plt)
    plt.plot(x_plt, y_plt4, KColor[3], label=Klegend[3])

    # Baseline
    #y_plt5 = MSE_baseline_dB_avg * np.ones(N_Epochs_plt)
    #plt.plot(x_plt, y_plt5, KColor[4], label=Klegend[4])

    plt.legend()
    plt.xlabel('Number of Training Epochs', fontsize=16)
    plt.ylabel('MSE Loss Value [dB]', fontsize=16)
    plt.title('MSE Loss [dB] - per Epoch', fontsize=16)
    plt.savefig(file_path + 'plt_model_test_dB')

    ####################
    ### dB Histogram ###
    ####################

    plt.figure(figsize=(50, 20))
    sns.distplot(10 * np.log10(MSE_test_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color='g', label = 'KalmanNet')
    sns.distplot(10 * np.log10(MSE_KF_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'b', label = 'Extended Kalman Filter')
    plt.legend()
    plt.title("Histogram [dB]")
    plt.savefig(file_path + 'plt_hist_dB')

    print('End')

def NNPlot_train_partial(MSE_KF_linear_arr_true, MSE_KF_dB_avg_true, 
                        MSE_KF_linear_arr_partial, MSE_KF_dB_avg_partial, 
                        MSE_test_linear_arr, MSE_test_dB_avg,
                        MSE_cv_dB_epoch, MSE_train_dB_epoch, MSE_baseline_dB_avg, file_path):

    N_Epochs_plt = np.shape(MSE_cv_dB_epoch)[0]

    ###########################
    ### Plot per epoch [dB] ###
    ###########################
    plt.figure(figsize = (50, 20))

    x_plt = range(0, N_Epochs_plt)

    # Train
    y_plt1 = MSE_train_dB_epoch[range(0, N_Epochs_plt)]
    plt.plot(x_plt, y_plt1, KColor_partial[0], label=Klegend_partial[0])

    # CV
    y_plt2 = MSE_cv_dB_epoch[range(0, N_Epochs_plt)]
    plt.plot(x_plt, y_plt2, KColor_partial[1], label=Klegend_partial[1])

    # KNet - Test
    y_plt3 = MSE_test_dB_avg * np.ones(N_Epochs_plt)
    plt.plot(x_plt, y_plt3, KColor_partial[2], label=Klegend_partial[2])

    # EKF_true
    y_plt4 = MSE_KF_dB_avg_true * np.ones(N_Epochs_plt)
    plt.plot(x_plt, y_plt4, KColor_partial[3], label=Klegend_partial[3])

    # Baseline
    #y_plt5 = MSE_baseline_dB_avg * np.ones(N_Epochs_plt)
    #plt.plot(x_plt, y_plt5, KColor[4], label=Klegend[4])

    # EKF_partial
    y_plt6 = MSE_KF_dB_avg_partial * np.ones(N_Epochs_plt)
    plt.plot(x_plt, y_plt6, KColor_partial[5], label=Klegend_partial[5])

    plt.legend()
    plt.xlabel('Number of Training Epochs', fontsize=16)
    plt.ylabel('MSE Loss Value [dB]', fontsize=16)
    plt.title('MSE Loss [dB] - per Epoch', fontsize=16)
    plt.savefig(file_path + 'plt_model_test_dB')

    ####################
    ### dB Histogram ###
    ####################

    plt.figure(figsize=(50, 20))
    sns.distplot(10 * np.log10(MSE_test_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color='g', label = 'KalmanNet')
    sns.distplot(10 * np.log10(MSE_KF_linear_arr_true), hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'b', label = 'Extended Kalman Filter True')
    sns.distplot(10 * np.log10(MSE_KF_linear_arr_partial), hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'r', label = 'Extended Kalman Filter Partial')
    plt.legend()
    plt.title("Histogram [dB]")
    plt.savefig(file_path + 'plt_hist_dB')

    print('End')

def KNPlot_test(MSE_KF_design_linear_arr, MSE_KF_data_linear_arr, MSE_KN_linear_arr):

    ####################
    ### dB Histogram ###
    ####################
    plt.figure(figsize=(50, 20))
    sns.distplot(10 * np.log10(MSE_KN_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color='g', label = 'KalmanNet')
    sns.distplot(10 * np.log10(MSE_KF_design_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'b', label = 'Extended Kalman Filter - design')
    sns.distplot(10 * np.log10(MSE_KF_data_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'r', label = 'Extended Kalman Filter - data')

    plt.title("Histogram [dB]")
    plt.savefig('plt_hist_dB_0')


    KF_design_MSE_mean_dB = 10 * np.log10(np.mean(MSE_KF_design_linear_arr))
    KF_design_MSE_median_dB = 10 * np.log10(np.median(MSE_KF_design_linear_arr))
    KF_design_MSE_std_dB = 10 * np.log10(np.std(MSE_KF_design_linear_arr))
    print("Extended Kalman Filter - Design:",
          "MSE - mean", KF_design_MSE_mean_dB, "[dB]",
          "MSE - median", KF_design_MSE_median_dB, "[dB]",
          "MSE - std", KF_design_MSE_std_dB, "[dB]")

    KF_data_MSE_mean_dB = 10 * np.log10(np.mean(MSE_KF_data_linear_arr))
    KF_data_MSE_median_dB = 10 * np.log10(np.median(MSE_KF_data_linear_arr))
    KF_data_MSE_std_dB = 10 * np.log10(np.std(MSE_KF_data_linear_arr))
    print("Extended Kalman Filter - Data:",
          "MSE - mean", KF_data_MSE_mean_dB, "[dB]",
          "MSE - median", KF_data_MSE_median_dB, "[dB]",
          "MSE - std", KF_data_MSE_std_dB, "[dB]")

    KN_MSE_mean_dB = 10 * np.log10(np.mean(MSE_KN_linear_arr))
    KN_MSE_median_dB = 10 * np.log10(np.median(MSE_KN_linear_arr))
    KN_MSE_std_dB = 10 * np.log10(np.std(MSE_KN_linear_arr))

    print("kalman Net:",
          "MSE - mean", KN_MSE_mean_dB, "[dB]",
          "MSE - median", KN_MSE_median_dB, "[dB]",
          "MSE - std", KN_MSE_std_dB, "[dB]")

def KFPlot(res_grid):

    plt.figure(figsize = (50, 20))
    x_plt = [-6, 0, 6]

    plt.plot(x_plt, res_grid[0][:], 'xg', label='minus')
    plt.plot(x_plt, res_grid[1][:], 'ob', label='base')
    plt.plot(x_plt, res_grid[2][:], '+r', label='plus')
    plt.plot(x_plt, res_grid[3][:], 'oy', label='base NN')

    plt.legend()
    plt.xlabel('Noise', fontsize=16)
    plt.ylabel('MSE Loss Value [dB]', fontsize=16)
    plt.title('Change', fontsize=16)
    plt.savefig('plt_grid_dB')

    print("\ndistribution 1")
    print("Kalman Filter")
    print(res_grid[0][0], "[dB]", res_grid[1][0], "[dB]", res_grid[2][0], "[dB]")
    print(res_grid[1][0] - res_grid[0][0], "[dB]", res_grid[2][0] - res_grid[1][0], "[dB]")
    print("KalmanNet", res_grid[3][0], "[dB]", "KalmanNet Diff", res_grid[3][0] - res_grid[1][0], "[dB]")

    print("\ndistribution 2")
    print("Kalman Filter")
    print(res_grid[0][1], "[dB]", res_grid[1][1], "[dB]", res_grid[2][1], "[dB]")
    print(res_grid[1][1] - res_grid[0][1], "[dB]", res_grid[2][1] - res_grid[1][1], "[dB]")
    print("KalmanNet", res_grid[3][1], "[dB]", "KalmanNet Diff", res_grid[3][1] - res_grid[1][1], "[dB]")

    print("\ndistribution 3")
    print("Kalman Filter")
    print(res_grid[0][2], "[dB]", res_grid[1][2], "[dB]", res_grid[2][2], "[dB]")
    print(res_grid[1][2] - res_grid[0][2], "[dB]", res_grid[2][2] - res_grid[1][2], "[dB]")
    print("KalmanNet", res_grid[3][2], "[dB]", "KalmanNet Diff", res_grid[3][2] - res_grid[1][2], "[dB]")

def NNPlot_test(MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg,
           MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg):

    N_Epochs_plt = 100

    ###############################
    ### Plot per epoch [linear] ###
    ###############################
    plt.figure(figsize = (50, 20))

    x_plt = range(0, N_Epochs_plt)

    # KNet - Test
    y_plt3 = MSE_test_linear_avg * np.ones(N_Epochs_plt)
    plt.plot(x_plt, y_plt3, KColor[2], label=Klegend[2])

    # KF
    y_plt4 = MSE_KF_linear_avg * np.ones(N_Epochs_plt)
    plt.plot(x_plt, y_plt4, KColor[3], label=Klegend[3])

    plt.legend()
    plt.xlabel('Number of Training Epochs', fontsize=16)
    plt.ylabel('MSE Loss Value [linear]', fontsize=16)
    plt.title('MSE Loss [linear] - per Epoch', fontsize=16)
    plt.savefig('plt_model_test_linear')

    ###########################
    ### Plot per epoch [dB] ###
    ###########################
    plt.figure(figsize = (50, 20))

    x_plt = range(0, N_Epochs_plt)

    # KNet - Test
    y_plt3 = MSE_test_dB_avg * np.ones(N_Epochs_plt)
    plt.plot(x_plt, y_plt3, KColor[2], label=Klegend[2])

    # KF
    y_plt4 = MSE_KF_dB_avg * np.ones(N_Epochs_plt)
    plt.plot(x_plt, y_plt4, KColor[3], label=Klegend[3])

    plt.legend()
    plt.xlabel('Number of Training Epochs', fontsize=16)
    plt.ylabel('MSE Loss Value [dB]', fontsize=16)
    plt.title('MSE Loss [dB] - per Epoch', fontsize=16)
    plt.savefig('plt_model_test_dB')

    ########################
    ### Linear Histogram ###
    ########################
    plt.figure(figsize=(50, 20))
    sns.distplot(MSE_test_linear_arr, hist=False, kde=True, kde_kws={'linewidth': 3}, color='g', label = 'KalmanNet')
    sns.distplot(MSE_KF_linear_arr, hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'b', label = 'Kalman Filter')
    plt.title("Histogram [Linear]")
    plt.savefig('plt_hist_linear')

    fig, axes = plt.subplots(2, 1, figsize=(50, 20), sharey=True, dpi=100)
    sns.distplot(MSE_test_linear_arr, hist=False, kde=True, kde_kws={'linewidth': 3}, color='g', label='KalmanNet', ax=axes[0])
    sns.distplot(MSE_KF_linear_arr, hist=False, kde=True, kde_kws={'linewidth': 3}, color='b', label='Kalman Filter', ax=axes[1])
    plt.title("Histogram [Linear]")
    plt.savefig('plt_hist_linear_1')

    ####################
    ### dB Histogram ###
    ####################

    plt.figure(figsize=(50, 20))
    sns.distplot(10 * np.log10(MSE_test_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color='g', label = 'KalmanNet')
    sns.distplot(10 * np.log10(MSE_KF_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'b', label = 'Kalman Filter')
    plt.title("Histogram [dB]")
    plt.savefig('plt_hist_dB')

    print('End')

def plotTrajectories(inputs, dim, titles, file_name):


    fig = plt.figure(figsize=(15,10))
    plt.Axes (fig, [0,0,1,1])
    #plt.subplots_adjust(wspace=-0.2, hspace=-0.2)
    matrix_size = int(np.ceil(np.sqrt(len(inputs))))
    #gs1 = gridspec.GridSpec(matrix_size,matrix_size)
    gs1 = gridspec.GridSpec(2,2, figure=fig)
    gs1.update(wspace=0, hspace=0)
    plt.rcParams["figure.frameon"] = False
    plt.rcParams["figure.constrained_layout.use"]= True
    i=0
    for title in titles:
        inputs_numpy = inputs[i].cpu().detach().numpy()
        gs1.update(wspace=-0.5,hspace=-0.3)
        if(dim==3):
            plt.rcParams["figure.frameon"] = False
            if(i<3):
                ax = fig.add_subplot(gs1[i],projection='3d')
            else:
                ax = fig.add_subplot(gs1[i:i+2],projection='3d')

            y_al = 0.73
            if(title == "True Trajectory"):
                c = 'k'
            elif(title == "Observation"):
                c = 'r'
            elif(title == "Extended Kalman\nFilter"):
                c = 'b'
                y_al = 0.65
            elif(title == "KalmanNet"):
                c = 'g'
            else:
                c = 'm'

            ax.set_axis_off()
            ax.set_title(title, y=y_al, fontdict={'fontsize': 20,'fontweight' : 20,'verticalalignment': 'baseline'})
            ax.plot(inputs_numpy[0,:], inputs_numpy[1,:], inputs_numpy[2,:], c, linewidth=1)

            ## Plot display 
            #ax.set_yticklabels([])
            #ax.set_xticklabels([])
            #ax.set_zticklabels([])
            #ax.set_xlabel('x')
            #ax.set_ylabel('y')
            #ax.set_zlabel('z')

        if(dim==2):
            ax = fig.add_subplot(matrix_size, matrix_size,i+1)
            ax.plot(np.arange(np.size(inputs_numpy[:],axis=1)), inputs_numpy[:], 'b', linewidth=0.75)
            ax.set_xlabel('time')
            ax.set_ylabel('x')
            ax.set_title(title, pad=10, fontdict={'fontsize': 20,'fontweight' : 20,'verticalalignment': 'baseline'})

        if(dim==4):
            ax = fig.add_subplot(matrix_size, matrix_size,i+1)
            print(inputs_numpy[0,0,:])
            ax.plot(np.arange(np.size(inputs_numpy[0,:],axis=1)), inputs_numpy[0,0,:], 'b', linewidth=0.75)
            ax.set_xlabel('time [s]')
            ax.set_ylabel('theta [rad]')
            ax.set_title(title, pad=10, fontdict={'fontsize': 20,'fontweight' : 20,'verticalalignment': 'baseline'})

        i +=1
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0, dpi=1000)