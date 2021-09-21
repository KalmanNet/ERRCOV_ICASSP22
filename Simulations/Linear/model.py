import autograd.numpy as np
from autograd import grad, jacobian
import torch
from torch import autograd

from parameters import F, F_mod, H, H_mod, m, n


def f(x):
    return torch.matmul(F, x)

def h(x):
    return torch.matmul(H, x)

def fInacc(x):
    return torch.matmul(F_mod, x)

def hInacc(x):
    return torch.matmul(H_mod, x)

def getJacobian(x, f):
    
    if(x.size()[1] == 1):
        y = torch.reshape((x.T),[x.size()[0]])
    
    Jac = autograd.functional.jacobian(f, y)
    Jac = Jac.view(-1,m)
    return Jac

"""
def getJacobian(x, a):
    
    if(x.size()[1] == 1):
        y = torch.reshape((x.T),[x.size()[0]])

    if(a == 'ObsAcc'):
        g = h
    elif(a == 'ModAcc'):
        g = f
    elif(a == 'ObsInacc'):
        g = hInacc
    elif(a == 'ModInacc'):
        g = fInacc

    return autograd.functional.jacobian(g, y)
"""
