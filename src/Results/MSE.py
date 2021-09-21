import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

import sys
sys.path.insert(1, 'src/')
from KalmanNet_plt import NNPlot_train, NNPlot_train_partial
np.set_printoptions(threshold=sys.maxsize)

class MSE:

    def __init__(self, session_path, partial=False, only_testing=False):

        self.session_path = session_path 

        file_results_read = torch.load(self.session_path + 'Results/results.pt')
        #file_data = torch.load(self.session_path + 'data_gen.pt')

        if(partial):
            [self.MSE_EKF_linear_arr_true, self.MSE_EKF_linear_avg_true, self.MSE_EKF_dB_avg_true, self.EKF_KG_array_true,_] = file_results_read["EKF True"]    

        [self.MSE_EKF_linear_arr, self.MSE_EKF_linear_avg, self.MSE_EKF_dB_avg, self.EKF_KG_array,_] = file_results_read["EKF"]


        [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch, self.MSE_train_linear_epoch, self.MSE_train_dB_epoch] = file_results_read["KNet_training"]
        [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, self.MSE_test_baseline_dB_avg, self.KNet_KG_array,_] = file_results_read["KNet_testing"]

        #[self.train_input, self.train_target, self.cv_input, self.cv_target, self.test_input, self.test_target] = file_data["All Data"]


    def writeMSE(self,Partial = False):
        file_results_write = open(self.session_path + '/Results/results.txt', "w")
        file_results_write.write("MSE Results\n")
        file_results_write.write("\nMSE-Baseline [dB]:" + str(self.MSE_test_baseline_dB_avg))
        file_results_write.write("\nMSE-EKF Test [dB]:" + str(self.MSE_EKF_dB_avg))
        file_results_write.write("\nMSE-KNet Test [dB]:" + str(self.MSE_test_dB_avg))

        if(Partial):
            file_results_write.write("\nMSE-EKF Full Test [dB]:" + str(self.MSE_EKF_dB_avg_true))
        file_results_write.close()



    def plotMSE(self, Partial=False):

        if (Partial):
            NNPlot_train_partial(self.MSE_EKF_linear_arr_true, self.MSE_EKF_dB_avg_true, 
                                self.MSE_EKF_linear_arr, self.MSE_EKF_dB_avg, 
                                self.MSE_test_linear_arr, self.MSE_test_dB_avg,
                                self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch, self.MSE_test_baseline_dB_avg, self.session_path +'/Plots/')
                        
        else:
            NNPlot_train(self.MSE_EKF_linear_arr, self.MSE_EKF_dB_avg, self.MSE_test_linear_arr, self.MSE_test_dB_avg,
                        self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch, self.MSE_test_baseline_dB_avg, self.session_path+'/Plots/')

    def writeParameters(self, NN_parameters, model_parameters, initial_conditions, model_parameters_true=None, initial_conditions_true=None, Partial=False):
        file_parameters_write = open(self.session_path + '/Results/parameters.txt', "w")
        file_parameters_write.write("NN parameters\n")
        file_parameters_write.write("[learning_rate, N_CV, N_Epochs, N_T, N_E, N_B, wd, nGRU]:" + str(NN_parameters))
        file_parameters_write.write("\n\nModel parameters")
        file_parameters_write.write("\n[q, r, T, T_test J, delta_t]:" + str(model_parameters))
        #file_parameters_write.write("[tau, g, L, lambda_r, lambda_q, delta_t, T]" + str(model_parameters))
        file_parameters_write.write("\n\nInitial Conditions")
        file_parameters_write.write("\n[m1x_0, m1x_0_test, m2x_0]:" + str(initial_conditions))

        if(Partial):
            file_parameters_write.write("\n\nModel parameters True")
            file_parameters_write.write("\n[q, r, T, T_test J, delta_t,H rotation[degrees]]:" + str(model_parameters_true))
            #file_parameters_write.write("[tau, g, L, lambda_r, lambda_q, delta_t, T]" + str(model_parameters))
            file_parameters_write.write("\n\nInitial Conditions")
            file_parameters_write.write("\n[m1x_0, m1x_0_test, m2x_0]:" + str(initial_conditions_true))

        now = datetime.now()
        file_parameters_write.write("\n\nSimulation Time: " + now.strftime("%d/%m/%Y %H:%M:%S"))
        file_parameters_write.close()

    def writeEpochs(self):
        file_results_write = open(self.session_path + '/Results/results_epochs.txt', "w")
        file_results_write.write("MSE FOR IN ALONG EPOCHS\n\n")
        file_results_write.write("\nMSE-Training [dB]:\n" + str(self.MSE_train_dB_epoch))
        file_results_write.write("\n\nMSE-CV [dB]:\n" + str(self.MSE_cv_dB_epoch))
        file_results_write.close()
        


'''
path_model = 'Simulations/Welling Models/Lorenz Atractor/'
path_session = path_model + 'Experiment 1: Full Info/r=0.2/q_e0/'

MSE_loader = MSE(path_session)
MSE_loader.writeEpochs()
'''