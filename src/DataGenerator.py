import math
from KalmanNet_data import DataGen, DecimateData, Decimate_and_perturbate_Data
import KalmanNet_sysmdl
import threading
import torch

import os

from filing_paths import path_model

import sys
sys.path.insert(1, path_model)

from parameters import Q_design, R_design, Q_mod, R_mod, T_test, J, J_mod, delta_t_gen, ratio
from parameters import m1x_0_design, m2x_0_design, m1x_0_mod, m2x_0_mod, T, lambda_q, lambda_r
from parameters import m1x_0_design_test, m1x_0_mod_test, T_mod, T_test_mod, roll_deg, delta_t_mod, m, n
from model import f_gen, f, h, h_rotated, h_nonlinear

# obs model can be either identity, rotated or nonlinear
# noise is 1/r^2 [dB]
def load_data(obs_model, noise, process_noise=None, discrete=False, randomize_init_conditions=True):

    # defining paths to save data
    if process_noise is None:
        pn_string = "none"
    else:
        pn_string = str(process_noise)

    if discrete and randomize_init_conditions:
        path_base = f"Simulations/Lorenz_Atractor/data/discrete_randomized/q_{pn_string}/"
    elif discrete and (not randomize_init_conditions):
        path_base = f"Simulations/Lorenz_Atractor/data/discrete_non_randomized/q_{pn_string}/"
    elif (not discrete) and randomize_init_conditions:
        path_base = f"Simulations/Lorenz_Atractor/data/continuous_randomized/q_{pn_string}/"
    elif (not discrete) and (not randomize_init_conditions):
        path_base = f"Simulations/Lorenz_Atractor/data/continuous_non_randomized/q_{pn_string}/"

    os.makedirs(path_base, exist_ok=True)

    IC_path = path_base + "IC.pt"
    obs_path = path_base + f"obs_{obs_model}_{str(noise)}.pt"
    
    if discrete:
        GT_path = path_base + "GT.pt"
    else:
        GT_test_path = path_base + "GT_test.pt"
        GT_test_long_path = path_base + "GT_test_long.pt"
        GT_test_short_path = path_base + "GT_test_short.pt"
        GT_CV_path = path_base + "GT_CV.pt"
        GT_train_path = path_base + "GT_train.pt"
        GT_undecimated_path = path_base + "GT_undecimated.pt"


    # defining lengths and numbers of trajectores
    # T * N needs to be an integer multiple of T_trajectory
    T_trajectory = int(T*ratio)
    #T_trajectory = 1000
    #T = int(T_trajectory / ratio)

    T_train = 100
    N_train = 360

    T_CV = T_train
    N_CV = 30

    T_test_short = 30
    T_test = 100
    T_test_long = 3000

    N_test_short = 100
    N_test = 30
    N_test_long = 1

    N_trajectories = math.ceil((T_train * N_train + T_CV * N_CV + 
            T_test_short * N_test_short + T_test * N_test + 
            T_test_long * N_test_long) / T_trajectory)

    # function to generate trajectories in the discrete case
    def gen_discrete(N, T):
        if process_noise is None:
            Q = Q_design
        else:
            Q = (10 ** (process_noise / 10)) * torch.eye(m)
        GT = torch.empty(N, m, T)
        sys_model = KalmanNet_sysmdl.SystemModel(f, Q, h, R_design, T)
        IC = torch.empty(N, m)
        for k in range(N):
            if randomize_init_conditions:
                m1x_0 = 5 * (2 * torch.rand(m1x_0_design.size()) - torch.ones(m1x_0_design.size()))
            else:
                m1x_0 = m1x_0_design
                
            IC[k, :] = torch.squeeze(m1x_0)
            sys_model.InitSequence(m1x_0, m2x_0_design)
            _,data = DataGen(sys_model, only_1_sequence=True)
            GT[k, :, :] = data
        return [GT, IC]

    if discrete:
        try:
            [GT_test, GT_test_long, GT_test_short, GT_train, GT_CV, IC_test, IC_test_long, IC_test_short, IC_train, IC_CV] = torch.load(GT_path)
        except:
            print("no GT found. generating new one")
            [GT_test, IC_test] = gen_discrete(N_test, T_test)
            [GT_test_long, IC_test_long] = gen_discrete(N_test_long, T_test_long)
            [GT_test_short, IC_test_short] = gen_discrete(N_test_short, T_test_short)
            [GT_CV, IC_CV] = gen_discrete(N_CV, T_CV)
            [GT_train, IC_train] = gen_discrete(N_train, T_train)
            GT_data = [GT_test, GT_test_long, GT_test_short, GT_train, GT_CV, IC_test, IC_test_long, IC_test_short, IC_train, IC_CV]
            torch.save(GT_data, GT_path)
    else:
        try:
            GT_test = torch.load(GT_test_path)
            GT_test_long = torch.load(GT_test_long_path)
            GT_test_short = torch.load(GT_test_short_path)
            GT_CV = torch.load(GT_CV_path)
            GT_train = torch.load(GT_train_path)
            [IC_test, IC_test_long, IC_test_short, IC_train, IC_CV] = torch.load(IC_path)
        except:
            try: 
                [GT_test_undecimated, GT_test_long_undecimated, GT_test_short_undecimated, GT_train_undecimated, GT_CV_undecimated] = torch.load(GT_undecimated_path)
                [IC_test, IC_test_long, IC_test_short, IC_train, IC_CV] = torch.load(IC_path)
            except:
                print("no GT found. generating new one")
                GT_CV_undecimated = torch.empty(N_CV, m, int(T_CV/ratio))
                GT_train_undecimated = torch.empty(N_train, m, int(T_train/ratio))
                GT_test_undecimated = torch.empty(N_test, m, int(T_test/ratio))
                GT_test_long_undecimated = torch.empty(N_test_long, m, int(T_test_long/ratio))
                GT_test_short_undecimated = torch.empty(N_test_short, m, int(T_test_short/ratio))

                # generating initial conditions
                def get_IC(N, randomize):
                    IC = torch.empty(N, m)
                    for k in range(N):
                        if randomize:
                            m1x_0 = 5 * (2 * torch.rand(m1x_0_design.size()) - torch.ones(m1x_0_design.size()))
                        else:
                            m1x_0 = m1x_0_design
                        IC[k, :] = torch.squeeze(m1x_0)
                    return IC

                IC_CV = get_IC(N_CV, randomize_init_conditions)
                IC_train = get_IC(N_train, randomize_init_conditions)
                IC_test = get_IC(N_test, randomize_init_conditions)
                IC_test_long = get_IC(N_test_long, False)
                IC_test_short = get_IC(N_test_short, randomize_init_conditions)

                if process_noise is None:
                    Q = Q_design
                else:
                    Q = ratio * (10 ** (process_noise / 10)) * torch.eye(m)

                # initializeing all the system models
                sys_model_CV = KalmanNet_sysmdl.SystemModel(f_gen, Q, h, R_design, int(T_CV/ratio))
                sys_model_train = KalmanNet_sysmdl.SystemModel(f_gen, Q, h, R_design, int(T_train/ratio))
                sys_model_test = KalmanNet_sysmdl.SystemModel(f_gen, Q, h, R_design, int(T_test/ratio))
                sys_model_test_long = KalmanNet_sysmdl.SystemModel(f_gen, Q, h, R_design, int(T_test_long/ratio))
                sys_model_test_short = KalmanNet_sysmdl.SystemModel(f_gen, Q, h, R_design, int(T_test_short/ratio))

                threads = 4*[None]

                #starting threads for the ground truth trajectories
                threads[0] = threading.Thread(target=gen_decimated, args=(0, N_test_long, 1, sys_model_test_long, IC_test_long, GT_test_long_undecimated,))
                threads[1] = threading.Thread(target=gen_decimated, args=(1, N_test_short, 1, sys_model_test_short, IC_test_short, GT_test_short_undecimated,))
                threads[2] = threading.Thread(target=gen_decimated, args=(2, N_test, 1, sys_model_test, IC_test, GT_test_undecimated,))
                threads[3] = threading.Thread(target=gen_decimated, args=(3, N_CV, 1, sys_model_CV, IC_CV, GT_CV_undecimated,))

                for t, thread in enumerate(threads):
                    print(f"starting thread {t}")
                    thread.start()
                for t in threads:
                    thread.join()

                GT_data_undecimated = [GT_test_undecimated, GT_test_long_undecimated, GT_test_short_undecimated, GT_train_undecimated, GT_CV_undecimated, IC_test, IC_test_long, IC_test_short, IC_train, IC_CV]
                torch.save(GT_data_undecimated, GT_undecimated_path)

                for t in range(4):
                    threads[t] = threading.Thread(target=gen_decimated, args=(t, N_train, 4, sys_model_train, IC_train, GT_train_undecimated,))

                for thread in threads:
                    thread.start()
                for t in threads:
                    thread.join()

                GT_data_undecimated = [GT_test_undecimated, GT_test_long_undecimated, GT_test_short_undecimated, GT_train_undecimated, GT_CV_undecimated]
                torch.save(GT_data_undecimated, GT_undecimated_path)
                IC_data = [IC_test, IC_test_long, IC_test_short, IC_train, IC_CV]
                torch.save(IC_data, IC_path)

            # decimating the long trajectories
            def decimate(GT):
                dec = GT[:, :, 0::int(1/ratio)]
                return dec

            GT_test = decimate(GT_test_undecimated)
            GT_test_long = decimate(GT_test_long_undecimated)
            GT_test_short = decimate(GT_test_short_undecimated)
            GT_CV = decimate(GT_CV_undecimated)
            GT_train = decimate(GT_train_undecimated)

            torch.save(GT_test, GT_test_path)
            torch.save(GT_test_long, GT_test_long_path)
            torch.save(GT_test_short, GT_test_short_path)
            torch.save(GT_CV, GT_CV_path)
            torch.save(GT_train, GT_train_path)



    # generating noisy observations
    def gen_obs(N, T, GT):
        # choosing appropriate observation model
        if obs_model == "identity":
            obs_noisefree = GT
        elif obs_model == "rotated":
            h_gen = h_rotated
        elif obs_model == "nonlinear":
            h_gen = h_nonlinear
        else:
            print(f"{obs_model} is not a valid argument for the observation model.\nPlease choose either identity, rotated or nonlinear")
        
        if obs_model == "rotated" or obs_model == "nonlinear":
            obs_noisefree = torch.empty(N_trajectories, n, T_trajectory)
            for k in range(N):
                for t in range(T):
                    obs_noisefree[k, :, t] = torch.squeeze(h_gen(GT[k, :, t]))

        r = 10 ** (-noise / 10)
        obs = obs_noisefree + r * torch.randn_like(obs_noisefree)
        return obs
     
    try:
        [obs_test, obs_test_long, obs_test_short, obs_train, obs_CV] = torch.load(obs_path)
    except:
        obs_test = gen_obs(N_test, T_test, GT_test)
        obs_test_long = gen_obs(N_test_long, T_test_long, GT_test_long)
        obs_test_short = gen_obs(N_test_short, T_test_short, GT_test_short)
        obs_train = gen_obs(N_train, T_train, GT_train)
        obs_CV = gen_obs(N_CV, T_CV, GT_CV)
        obs_data = [obs_test, obs_test_long, obs_test_short, obs_train, obs_CV]
        torch.save(obs_data, obs_path)

    return [GT_test, obs_test, IC_test,
            GT_test_long, obs_test_long, IC_test_long,
            GT_test_short, obs_test_short, IC_test_short,
            GT_train, obs_train, IC_train,
            GT_CV, obs_CV, IC_CV]

# function for generating trajectories with multiple threads
def gen_decimated(t, N_trajectories, N_threads, sys_model, IC, GT):
    if N_threads > 1:
        N_gen = round(math.floor(N_trajectories/N_threads))
        if N_trajectories % N_threads < t:
            N_gen += 1
    else:
        N_gen = N_trajectories
    for k in range(N_gen):
        if N_threads > 1:
            idx = t + N_threads * k
        else:
            idx = k
        if idx < N_trajectories:
            print(f"starting index {idx}!")
            m1x_0 = torch.reshape(IC[idx, :], m1x_0_design.shape)
            sys_model.InitSequence(m1x_0, m2x_0_design)
            _,data = DataGen(sys_model, only_1_sequence=True)
            GT[idx, :, :] = data[:, :, :]
            print(f"index {idx} done!")


# old data generator for generating chopped trajectories
def load_data_old(obs_model, noise, process_noise=None, discrete=False, randomize_init_conditions=True):

    if randomize_init_conditions == True:
        if not discrete:
            if process_noise is None:
                GT_path = "data/GT.pt"
                GT_undecimated_path = "data/GT_undecimated.pt"
                obs_path = "data/" + obs_model + "/" + obs_model + str(noise) + ".pt"
                obs_noisefree_path = "data/" + obs_model + "/" + obs_model + "_noisefree.pt"  
            else:
                GT_path = f"data/GT_{str(process_noise)}dB.pt"
                GT_undecimated_path = f"data/GT_undecimated_{str(process_noise)}dB.pt"
                obs_path = "data/" + obs_model + "/" + obs_model + str(noise) + "_" + str(process_noise) + ".pt"
                obs_noisefree_path = "data/" + obs_model + "/" + obs_model + "_noisefree_" + str(process_noise) + ".pt"
        else:
            if process_noise is None:
                GT_path = "data/GT_discrete.pt"
                GT_undecimated_path = "data/GT_undecimated_discrete.pt"
                obs_path = "data/" + obs_model + "/" + obs_model + str(noise) + "_discrete.pt"
                obs_noisefree_path = "data/" + obs_model + "/" + obs_model + "_noisefree_discrete.pt"  
            else:
                GT_path = f"data/GT_{str(process_noise)}dB_discrete.pt"
                GT_undecimated_path = f"data/GT_undecimated_{str(process_noise)}dB_discrete.pt"
                obs_path = "data/" + obs_model + "/" + obs_model + str(noise) + "_" + str(process_noise) + "_discrete.pt"
                obs_noisefree_path = "data/" + obs_model + "/" + obs_model + "_noisefree_" + str(process_noise) + "_discrete.pt"
    else:
        if not discrete:
            if process_noise is None:
                GT_path = "data/unrandomized/GT.pt"
                GT_undecimated_path = "data/unrandomized/GT_undecimated.pt"
                obs_path = "data/unrandomized/" + obs_model + "/" + obs_model + str(noise) + ".pt"
                obs_noisefree_path = "data/unrandomized/" + obs_model + "/" + obs_model + "_noisefree.pt"  
            else:
                GT_path = f"data/unrandomized/GT_{str(process_noise)}dB.pt"
                GT_undecimated_path = f"data/unrandomized/GT_undecimated_{str(process_noise)}dB.pt"
                obs_path = "data/unrandomized/" + obs_model + "/" + obs_model + str(noise) + "_" + str(process_noise) + ".pt"
                obs_noisefree_path = "data/" + obs_model + "/" + obs_model + "_noisefree_" + str(process_noise) + ".pt"
        else:
            if process_noise is None:
                GT_path = "data/unrandomized/GT_discrete.pt"
                GT_undecimated_path = "data/unrandomized/GT_undecimated_discrete.pt"
                obs_path = "data/unrandomized/" + obs_model + "/" + obs_model + str(noise) + "_discrete.pt"
                obs_noisefree_path = "data/" + obs_model + "/" + obs_model + "_noisefree_discrete.pt"  
            else:
                GT_path = f"data/unrandomized/dt_02/GT_{str(process_noise)}dB_discrete.pt"
                GT_undecimated_path = f"data/unrandomized/dt_02/GT_undecimated_{str(process_noise)}dB_discrete.pt"
                obs_path = "data/unrandomized/dt_02/" + obs_model + "/" + obs_model + str(noise) + "_" + str(process_noise) + "_discrete.pt"
                obs_noisefree_path = "data/unrandomized/dt_02/" + obs_model + "/" + obs_model + "_noisefree_" + str(process_noise) + "_discrete.pt"

    if not discrete:
        # load or generate ground truth
        try:
            GT = torch.load(GT_path)
        except:
            print("No existing ground truth file found. Starting data generation")
            if process_noise is None:
                Q = Q_design
            else:
                Q = ratio * (10 ** (process_noise / 10)) * torch.eye(m)

            if randomize_init_conditions is False and process_noise is None:
                GT = torch.empty(1, m, T_trajectory)
                GT_undecimated = torch.empty(1, m, T)
                N_threads = 1

            else:
                GT = torch.empty(N_trajectories, m, T_trajectory)
                GT_undecimated = torch.empty(N_trajectories, m, T)
                N_threads = 4

            threads = N_threads * [None]

            for t in range(N_threads):
                threads[t] = threading.Thread(target=generate_trajectory, args=(t, N_trajectories, N_threads, GT, GT_undecimated, Q, randomize_init_conditions,))
                threads[t].start()
            for thread in threads:
                thread.join()
            if not randomize_init_conditions:
                GT = GT.repeat(N_trajectories, 1, 1)
            torch.save(GT, GT_path)
            torch.save(GT_undecimated, GT_undecimated_path)
            print("done!")
    else:
        try:
            GT = torch.load(GT_path)
        except:
            print("No existing ground truth file found. Starting data generation for discrete case")
            if process_noise is None:
                Q = Q_design
            else:
                Q = (10 ** (process_noise / 10)) * torch.eye(m)
            GT = torch.empty(N_trajectories, m, T_trajectory)
            sys_model = KalmanNet_sysmdl.SystemModel(f, Q, h, R_design, T_trajectory)
            for k in range(N_trajectories):
                print(f"generating trajectory no {k}")
                if randomize_init_conditions:
                    m1x = 2 * torch.rand(m1x_0_design.size()) - torch.ones(m1x_0_design.size())
                    sys_model.InitSequence(m1x, m2x_0_design)
                else:
                    sys_model.InitSequence(m1x_0_design, m2x_0_design)
                _, data = DataGen(sys_model, only_1_sequence=True)
                GT[k, :, :] = data
            torch.save(GT, GT_path)
    

    # choosing appropriate observation model
    if obs_model == "identity":
        h_gen = h
    elif obs_model == "rotated":
        h_gen = h_rotated
    elif obs_model == "nonlinear":
        h_gen = h_nonlinear
    else:
        print(f"{obs_model} is not a valid argument for the observation model.\nPlease choose either identity, rotated or nonlinear")
    
    # load noise free observations, generate from scratch if not available
    try:
        obs_noisefree = torch.load(obs_noisefree_path)
    except:
        print("No noise free observation found. Generating new observation")
        obs_noisefree = torch.empty(N_trajectories, n, T_trajectory)
        for k in range(N_trajectories):
            for t in range(T_trajectory):
                obs_noisefree[k, :, t] = torch.squeeze(h_gen(GT[k, :, t]))
        torch.save(obs_noisefree, obs_noisefree_path)
    

    # load noisy version of observation, generate noise if not available
    try:
        obs = torch.load(obs_path)
    except:
        print("No noisy observation found. Generating new observation")
        r = 10 ** (-noise / 10)
        obs = obs_noisefree + r * torch.randn_like(obs_noisefree)
        torch.save(obs, obs_path)


    # cut trajectories into length T
    def shorten_trajectories(data, T):
        data = torch.split(data, T, dim=2)
        return torch.cat(data, dim=0)


    # split data into training, validation and test sets
    def split(data):
        sizes = [int(N_train * T_train / T_trajectory),
                int(N_CV * T_CV / T_trajectory),
                int(N_test_short * T_test_short / T_trajectory),
                int(N_test * T_test / T_trajectory),
                int(N_test_long * T_test_long / T_trajectory)]
        train, CV, test_short, test, test_long = torch.split(data, sizes, dim=0)
        train = shorten_trajectories(train, T_train)
        CV = shorten_trajectories(CV, T_CV)
        test_short = shorten_trajectories(test_short, T_test_short)
        test = shorten_trajectories(test, T_test)
        test_long = shorten_trajectories(test_long, T_test_long)
        return [train, CV, test_short, test, test_long]

    [train_target, CV_target, test_target_short, test_target, test_target_long] = split(GT)
    [train_input, CV_input, test_input_short, test_input, test_input_long] = split(obs)

    return [train_target, train_input, CV_target, CV_input, 
            test_target_short, test_input_short, test_target, test_input,
            test_target_long, test_input_long]


                  
def generate_trajectory(t, N_trajectories, N_threads, GT, GT_undecimated, Q, randomize_init_conditions):
    sys_model = KalmanNet_sysmdl.SystemModel(f_gen, Q, h, R_design, T)
    N = round(math.floor(N_trajectories/N_threads))
    if N_trajectories % N_threads < t:
        N += 1
    for i in range(N):
        idx = t + math.ceil(N_trajectories/N_threads) * i
        if idx < N_trajectories:
            print(f"starting thread {t} generating trajectory with index {idx}")
            # the trajectories are initialized uniformly at random with x_0 in [-1, 1]^m 
            if randomize_init_conditions:
               m1x = 2 * torch.rand(m1x_0_design.size()) - torch.ones(m1x_0_design.size())
               sys_model.InitSequence(m1x, m2x_0_design)
            else:
                sys_model.InitSequence(m1x_0_design, m2x_0_design)
            _, data = DataGen(sys_model, only_1_sequence=True)
            GT[idx, :, :] = data[:, : ,0::int(1/ratio)]
            GT_undecimated[idx, :, :] = data[:, :, :]
            print(f"index {idx} done!")





