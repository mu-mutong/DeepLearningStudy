# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 10:20:07 2019

@author: 木沐童
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def loadDataSet(start,end):
    #data = np.loadtxt(fileName,delimiter='\t')
    df = np.loadtxt(open("D:\CS\Graduate\暑假学习\算法\Data\wine.data","rb"),delimiter=",",skiprows=0) 
    data = df[:,start:end]
    
    return data
mark = ['r', 'g', 'b', 'y', 'm']
dataSet = loadDataSet(0,4)
m,n = dataSet.shape
ax = plt.figure().add_subplot(111, projection = '3d')
for i in range(m):
    markIndex = int(dataSet[i,0])
    x = dataSet[i,1]
    y = dataSet[i,2]
    z = dataSet[i,3]
    ax.scatter(x,y,z,c = mark[markIndex])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label') 
  

plt.show()