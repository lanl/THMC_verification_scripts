#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt



def solvr(W, n):
    return [-1 * (n * W[0] + 12 * W[0]**2 * W[1]**2) / (4 * W[1]**3), W[0]]

# Next, define parameters and ICs
def main():
    a_n = np.arange(0, 10, 0.01) # values of similarity variable
    P_i = 21.9e6 # initial pressure
    P_0 = 21.0e6 # boundary pressure - note system in paper is "upside-down"
    b_i = 1e-5 # initial aperture
    kappa_n = 100000.0e6 # fracture stiffness
    w_0 = (1 - (P_i - P_0) / (b_i * kappa_n))
    print(w_0)
    Q_0 =  -0.13 # unknown? but for w_0 = 0.1, should be around .13 based on paper figure
    W_0_0 = (-1 * Q_0) / w_0**3
    #W_0_0 = -1*(Q_0 / w_0**3) # not clear which is correct
    W_1_0 = w_0
    asol = integrate.odeint(solvr, [W_0_0, W_1_0], a_n)
    
    astack = np.c_[a_n, asol[:,0], asol[:,1]] # last column is solution
    np.savetxt('approx.txt', astack)

if __name__ == '__main__':
    main()





