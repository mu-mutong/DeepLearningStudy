# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 10:08:42 2019

@author: 木沐童
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import random as nr
from LoadMinist import load_mnist as lm

train_image_path = 'train-images.idx3-ubyte'
train_label_path = 'train-labels.idx1-ubyte'
test_image_path = 't10k-images.idx3-ubyte'
test_label_path = 't10k-labels.idx1-ubyte'
(x,y),(tx,ty) = lm(train_image_path, train_label_path, test_image_path, test_label_path, normalize=True, one_hot=True)

alpha = 0.1
w1 = nr.randn(100,784)*0.01
w2 = nr.randn(10,100)*0.01
b1 = np.zeros((100,1))
b2 = np.zeros((10,1))

def testTrain(x,y,w1,w2,b1,b2):
    z1 = np.dot(w1,x.T)+b1
    a1 = 1/(1+np.exp(-1*z1))
    z2 = np.dot(w2,a1)+b2
    a2 = 1/(1+np.exp(-1*z2))
    y_ = a2.argmax(axis=0)
    
    y = y.argmax(axis=0)
    
    accuracy = np.sum(y_==y)   
    accuracy = accuracy *1.0/10000
    print(accuracy)

for i in range(2000):
    z1 = np.dot(w1,x.T)+b1
    a1 = 1/(1+np.exp(-1*z1))
    z2 = np.dot(w2,a1)+b2
    a2 = 1/(1+np.exp(-1*z2))
    
    L1 = -(y.T*np.log(a2)+(1-y.T)*np.log(1-a2)) 
    L1 = np.sum(L1)/60000
    L1 = np.sum(L1)/10
    plt.scatter(i,L1)
    if(i%100==0):
        print("L1 = %f"%(L1))
        
   
    
    dz2 = a2-y.T
    dw2 = np.dot(dz2,a1.T)/60000
    db2 = np.sum(dz2,axis = 1,keepdims = True)/60000
    
    s = np.dot(w2.T,dz2)
    v = a1*(1-a1)
    dz1 = s*v 
    dw1 = np.dot(dz1,x)/60000
    db1 = np.sum(dz1,axis = 1,keepdims = True)/60000
    
    w2 = w2-alpha*dw2
    b2 = b2-alpha*db2
    w1 = w1-alpha*dw1
    b1 = b1-alpha*db1
   
testTrain(tx,ty.T,w1,w2,b1,b2) 

