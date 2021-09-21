import numpy as np
import torch
import torch.nn as nn
import time

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    dev = torch.device("cpu")


from NN_parameters import N_T

def NNTest(SysModel, test_input, test_target, path_results, nclt=False, rnn=False, IC=None, calculate_covariance=False):

    N_T = test_input.size()[0]
    MSE_test_linear_arr = torch.empty([N_T])

    # MSE LOSS Function
    loss_fn = nn.MSELoss(reduction='mean')

    if(rnn):
        Model = torch.load(path_results+'best-model_rnn.pt')
    else:
        Model = torch.load(path_results+'best-model.pt')

    Model.eval()
    torch.no_grad()

    KGain_array = torch.zeros((SysModel.T, Model.m, Model.n))
    x_out_array = torch.empty(N_T, SysModel.m, SysModel.T)
    if calculate_covariance:
        P_array = torch.empty(N_T, SysModel.T, Model.n, Model.n)
    
    start = time.time()
    for j in range(0, N_T):
        Model.i = 0
        # Unrolling Forward Pass
        if nclt:
            Model.InitSequence(SysModel.m1x_0, SysModel.m2x_0, SysModel.T, calculate_covariance=calculate_covariance)
        elif IC is None:
            Model.InitSequence(torch.unsqueeze(test_target[j, :, 0], dim=1), SysModel.m2x_0, SysModel.T, calculate_covariance=calculate_covariance)
        else:
            init_cond = torch.reshape(IC[j, :], SysModel.m1x_0.shape)
            Model.InitSequence(init_cond, SysModel.m2x_0, SysModel.T, calculate_covariance=calculate_covariance)

        
        y_mdl_tst = test_input[j, :, :]

        x_Net_mdl_tst = torch.empty(SysModel.m, SysModel.T).to(dev, non_blocking=True)
        test_target = test_target.to(dev, non_blocking=True)
        
        for t in range(0, SysModel.T):
            x_Net_mdl_tst[:,t] = Model(y_mdl_tst[:,t])
        
        if(nclt):
            if x_Net_mdl_tst.size()[0] == 6:
                mask = torch.tensor([True,False,False,True,False,False])
            else:
                mask = torch.tensor([True,False,True,False])
            MSE_test_linear_arr[j] = loss_fn(x_Net_mdl_tst[mask], test_target[j, :, :]).item()
        else:
            MSE_test_linear_arr[j] = loss_fn(x_Net_mdl_tst, test_target[j, :, :]).item()
        x_out_array[j,:,:] = x_Net_mdl_tst

        try:
            KGain_array = torch.add(Model.KGain_array, KGain_array)
            KGain_array /= N_T
        except:
            KGain_array = None

        if calculate_covariance:
            P_array[j, :, :, :] = Model.P_array

    end = time.time()
    t = end - start

    # Average
    MSE_test_linear_avg = torch.mean(MSE_test_linear_arr)
    MSE_test_dB_avg = 10 * torch.log10(MSE_test_linear_avg)

    if calculate_covariance:
        return [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, KGain_array, x_out_array, P_array, t]
    else:
        return [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, KGain_array, x_out_array, t]
