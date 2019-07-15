# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 19:03:48 2019

@author: 木沐童
"""

import numpy as np
def load(start,end):    
    df = np.loadtxt(open("IRIS.txt","rb"),delimiter=",",skiprows=0) 
    x = df[:,start:end]
    return x
if __name__ == '__main__':
    train = load(0,2)