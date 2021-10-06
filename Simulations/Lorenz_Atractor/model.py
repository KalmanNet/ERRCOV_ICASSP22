import numpy as np
import math
import torch
from torch import autograd
from parameters import m, n, J, delta_t, delta_t_gen, H_design, B, C, B_mod, C_mod, delta_t_mod, J_mod, H_mod, H_design_inv, H_mod_inv, H_rotated, J_inacc

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    dev = torch.device("cpu")

def f_gen(x):

    #A = torch.add(torch.einsum('nhw,wa->nh', B, x).T,C)
    A = torch.add(torch.reshape(torch.matmul(B, x),(m,m)).T,C)#.to(dev, non_blocking=True)
    
    # Taylor Expansion for F    
    F = torch.eye(m)#.to(dev, non_blocking=True)
    for j in range(1,J+1):
        F_add = torch.matrix_power(A*delta_t_gen, j)/math.factorial(j)
        F = torch.add(F, F_add)

    return torch.matmul(F, x)


def f(x):

    #A = torch.add(torch.einsum('nhw,wa->nh', B, x).T,C)
    A = torch.add(torch.reshape(torch.matmul(B, x),(m,m)).T,C)#.to(dev, non_blocking=True)
    
    # Taylor Expansion for F    
    F = torch.eye(m)#.to(dev, non_blocking=True)
    for j in range(1,J+1):
        F_add = torch.matrix_power(A*delta_t, j)/math.factorial(j)
        F = torch.add(F, F_add)

    return torch.matmul(F, x)

def f_approx(x):
    #A = torch.add(torch.einsum('nhw,wa->nh', B, x).T,C)
    A = torch.add(torch.reshape(torch.matmul(B, x),(m,m)).T,C)#.to(dev, non_blocking=True)
    
    # Taylor Expansion for F    
    F = torch.eye(m)#.to(dev, non_blocking=True)
    for j in range(1,J_inacc+1):
        F_add = torch.matrix_power(A*delta_t, j)/math.factorial(j)
        F = torch.add(F, F_add)

    return torch.matmul(F, x)



def h(x):
    x = x#.to(dev, non_blocking=True)
    return torch.matmul(H_design,x)
    #return toSpherical(x)

def h_rotated(x):
    x = x#.to(dev, non_blocking=True)
    return torch.matmul(H_rotated,x)
    #return toSpherical(x)

def h_nonlinear(x):
    return toSpherical(x)


def f_interpolate(x, n=2):
    
    for _ in range(n):
        A = torch.add(torch.reshape(torch.matmul(B_mod, x),(m,m)).T,C_mod)#.to(dev, non_blocking=True)
   
        # Taylor Expansion for F    
        F = torch.eye(m)#.to(dev, non_blocking=True)
        for j in range(1,J+1):
            F_add = torch.matrix_power(A*delta_t/n, j)/math.factorial(j)
            F = torch.add(F, F_add)
        x = torch.matmul(F, x)
        return x

def f_interpolate_approx(x, n=2):

    for _ in range(n):
        A = torch.add(torch.reshape(torch.matmul(B_mod, x),(m,m)).T,C_mod)#.to(dev, non_blocking=True)
   
        # Taylor Expansion for F    
        F = torch.eye(m)#.to(dev, non_blocking=True)
        for j in range(1,J_inacc+1):
            F_add = torch.matrix_power(A*delta_t/n, j)/math.factorial(j)
            F = torch.add(F, F_add)
        x = torch.matmul(F, x)
    return x


def hInacc(x):
    return torch.matmul(H_mod,x)
    #return toSpherical(x)

"""
def getJacobian(x, a):
    
    if(x.size()[1] == 1):
        y = torch.reshape((x.T),[x.size()[0]])
    
    if(a == 'ObsAcc'):
        g = h
    elif(a == 'ModAcc'):
        g = f
    elif(a == 'ObsInacc'):
        g = h_rotated
    elif(a == 'ModInacc'):
        g = f_approx

    Jac = autograd.functional.jacobian(g, y)
    Jac = Jac.view(-1,m)
    return Jac

"""
def getJacobian(x, f):
    
    if(x.size()[1] == 1):
        y = torch.reshape((x.T),[x.size()[0]])
    
    Jac = autograd.functional.jacobian(f, y)
    Jac = Jac.view(-1,m)
    return Jac


def toSpherical(cart):

    rho = torch.norm(cart, p=2).view(1,1)
    phi = torch.atan2(cart[1, ...], cart[0, ...]).view(1, 1)
    phi = phi + (phi < 0).type_as(phi) * (2 * np.pi)

    theta = torch.acos(cart[2, ...] / rho).view(1, 1)

    spher = torch.cat([rho, theta, phi], dim=0)

    return spher

def toCartesian(sphe):

    rho = sphe[0,:]
    theta = sphe[1,:]
    phi = sphe[2,:]

    x = (rho * torch.sin(theta) * torch.cos(phi)).view(1,-1)
    y = (rho * torch.sin(theta) * torch.sin(phi)).view(1,-1)
    z = (rho * torch.cos(theta)).view(1,-1)

    cart = torch.cat([x,y,z],dim=0)

    return cart

def hInv(y):
    return torch.matmul(H_design_inv,y)
    #return toCartesian(y)


def hInaccInv(y):
    return torch.matmul(H_mod_inv,y)
    #return toCartesian(y)

'''
x = torch.tensor([[1],[1],[1]]).float() 
H = getJacobian(x, 'ObsAcc')
print(H)
print(h(x))

F = getJacobian(x, 'ModAcc')
print(F)
print(f(x))
'''
