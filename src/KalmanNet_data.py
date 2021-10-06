import torch
import math
import torch
import numpy as np

from KalmanNet_sysmdl import SystemModel
from filing_paths import path_model
from NN_parameters import N_E, N_CV, N_T

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    dev = torch.device("cpu")


def DataGen(SysModel_data, Only_test=False, Only_training=False, sequential_training=False, T_test=0, batch_perc = 1, only_1_sequence=False, repeat_test_trajectory=False):

    # batch_perc indicates the percentage of the total dataset size we are going to save in each batch
    N_E_reduced = math.floor(N_E * batch_perc)
    N_CV_reduced = math.floor(N_CV * batch_perc)
    N_T_reduced = math.floor(N_T * batch_perc)

    if(Only_test or only_1_sequence):
        ##############################
        ### Generate Test Sequence ###
        ##############################
        if(only_1_sequence):
            N_T_reduced = 1

        SysModel_data.GenerateBatch(N_T_reduced, 0.5, False, False)
        test_input = SysModel_data.Input
        test_target = SysModel_data.Target

        return[test_input, test_target]

    if(Only_training):
        ##################################
        ### Generate Training Sequence ###
        ##################################
        SysModel_data.GenerateBatch(N_E_reduced, 0.5, False, sequential_training, T_test)
        training_input = SysModel_data.Input
        training_target = SysModel_data.Target

        # Initialize CV for resembling to testing
        SysModel_data.InitSequence(SysModel_data.m1x_0, SysModel_data.m2x_0)

        ####################################
        ### Generate Validation Sequence ###
        ####################################
        SysModel_data.GenerateBatch(N_CV_reduced, 0.5, False, sequential_training, T_test)
        cv_input = SysModel_data.Input
        cv_target = SysModel_data.Target

        return [training_input, training_target, cv_input, cv_target]
    
    else:
        ##################################
        ### Generate Training Sequence ###
        ##################################
        SysModel_data.GenerateBatch(N_E_reduced, 0.5)
        training_input = SysModel_data.Input
        training_target = SysModel_data.Target

        ####################################
        ### Generate Validation Sequence ###
        ####################################
        SysModel_data.GenerateBatch(N_CV_reduced, 0.5)
        cv_input = SysModel_data.Input
        cv_target = SysModel_data.Target

        ##############################
        ### Generate Test Sequence ###
        ##############################
        if repeat_test_trajectory:
            SysModel_data.GenerateBatch(1, 0.5)
            process = SysModel_data.Target
            noisefree_obs = getObs(process, SysModel_data.h)
            noisefree_obs = torch.cat(N_T_reduced * [noisefree_obs])
            test_target = torch.cat(N_T_reduced * [process])
            noise = np.random.multivariate_normal(np.zeros(SysModel_data.n), SysModel_data.R.numpy(), size=(N_T_reduced, SysModel_data.T))
            noise = torch.tensor(noise.transpose(0,2,1)).to(torch.float)
            test_input = noisefree_obs + noise
        else:
            SysModel_data.GenerateBatch(N_T_reduced, 0.5)
            test_input = SysModel_data.Input
            test_target = SysModel_data.Target

        # return
        return [training_input, training_target, cv_input, cv_target, test_input, test_target]

def Data_Gen_multiple_obs(SysModel_data, N_trajectories, N_obs):

    SysModel_data.GenerateBatch(N_trajectories, 0.5)
    target = torch.cat(N_obs * [SysModel_data.Target])
    noisefree_obs = getObs(target, SysModel_data.h)
    noise = np.random.multivariate_normal(np.zeros(SysModel_data.n), SysModel_data.R.numpy(), size=(N_trajectories * N_obs, SysModel_data.T))
    noise = torch.tensor(noise.transpose(0,2,1)).to(torch.float)
    noisy_obs = noisefree_obs + noise

    return [noisy_obs, target]



def DecimateData(all_tensors, t_gen,t_mod, offset=0):

    # ratio: defines the relation between the sampling time of the true process and of the model (has to be an integer)
    ratio = round(t_mod/t_gen)

    print(ratio)
    i = 0
    all_tensors_out = all_tensors
    for tensor in all_tensors:
        tensor = tensor[:,(0+offset)::ratio]
        if(i==0):
            all_tensors_out = torch.cat([tensor], dim=0).view(1,all_tensors[0].size()[1],-1)
        else:
            all_tensors_out = torch.cat([all_tensors_out,tensor], dim=0)
        i += 1

    return all_tensors_out

def Decimate_and_perturbate_Data(true_process, delta_t, delta_t_mod, N_examples, h, lambda_r, offset=0):

    # Decimate high resolution process
    decimated_process = DecimateData(true_process, delta_t, delta_t_mod, offset)

    noise_free_obs = getObs(decimated_process,h)

    # Replicate for computation purposes
    decimated_process = torch.cat(int(N_examples)*[decimated_process])
    noise_free_obs = torch.cat(int(N_examples)*[noise_free_obs])


    # Observations; additive Gaussian Noise
    observations = noise_free_obs + torch.randn_like(decimated_process).to(dev, non_blocking=True) * lambda_r

    return [decimated_process, observations]

def getObs(sequences, h):
    i = 0
    sequences_out = torch.zeros_like(sequences)
    for sequence in sequences:
        for t in range(sequence.size()[1]):
            sequences_out[i,:,t] = h(sequence[:,t])
    i = i+1

    return sequences_out




'''
data_gen_file = path_session + "data_gen.pt"

# Generate Data with Real Model
load = True

print("Start Loading Data")
data_file = torch.load(data_gen_file)
[train_input, train_target, cv_input, cv_target, test_input, test_target] = data_file["All Data"]

T = 20
t_gen = 1
t_mod = 10
T_mod = math.ceil(T * t_gen/t_mod)

[train_input_d, train_target_d, cv_input_d, cv_target_d, test_input_d, test_target_d] = DecimateData(t_gen, t_mod, train_input, train_target, cv_input, cv_target, test_input, test_target)

print(train_input.size(), train_input_d.size())
print(train_target.size(), train_target_d.size())
print(cv_input.size(), cv_input_d.size())
print(cv_target.size(), cv_target_d.size())
print(test_input.size(), test_input_d.size())
print(test_target.size(), test_target_d.size())
print(T_mod)
'''
