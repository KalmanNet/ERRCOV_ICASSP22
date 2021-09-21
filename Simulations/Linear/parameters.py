import torch
import math

#########################
### Design Parameters ###
#########################
m = n = 1
theta = 1 * 2 * math.pi/360
T = 100

m1x_0 = torch.ones(m, 1)
m2x_0 = torch.zeros(m,m)


#######################
### True Parameters ###
#######################

if m == 1:
    F = torch.tensor([[.9]])
    H = torch.tensor([[1.]])
elif m == 2:
    F = torch.tensor([[1., 1.],
                        [0., 1.]])
    H = torch.tensor([[1., 1.],
                        [1., 0.]])

# Noise Parameters
sigma_q = 1
sigma_r = 1

# Noise Matrices
Q = (sigma_q**2) * torch.eye(n)
R = (sigma_r**2) * torch.eye(m)

########################
### Model Parameters ###
########################

if m == 1:
    F_mod = torch.tensor([[.5]])
    H_mod = torch.tensor([[1.]])
elif m == 2:
    rot = torch.tensor([[math.cos(theta), -math.sin(theta)],
                        [math.sin(theta), math.cos(theta)]])
    F_mod = torch.matmul(rot, F)
    H_mod = H

# Noise Parameters
sigma_q = 1
sigma_r = 1

# Noise Matrices
Q_mod = (sigma_q**2) * torch.eye(n)
R_mod = (sigma_r**2) * torch.eye(m)
