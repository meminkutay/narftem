# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 01:03:40 2021

@author: PC
"""
import numpy as np

def f_lin_interp1(x, y, xo):
# linear interpolation. Assumes linear incrrease from previous point to the next.    
    dxo     = x-xo        
    Idxo    = np.argwhere(dxo<=0)         
    n       = Idxo[-1]
    if np.all(y==0): # if only the layer no is needed
        yo = 0
    else:
        if n==np.shape(x)[0]-1:
            yo = y[-1]
            # print("True")            
        else:
            dy   = y[n] - y[n-1] 
            dx   = x[n] - x[n-1] 
            # print(dy,dx)
            yo = y[n-1] + dy/dx*(xo-x[n-1])

    return yo, n