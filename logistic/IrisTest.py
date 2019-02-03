# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 17:49:52 2019

@author: dellpc
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import random as nr
#读取训练数据
df = np.loadtxt(open("Iris.txt","rb"),delimiter=",",skiprows=0)  
#设置学习度为alpha= 0.8
#设置迭代次数为1000代
#x矩阵为Iris的四个长度，y矩阵为对应类别，因为本次实验为二元分类，故只用了Iris数据集前两种花
#的数据，其中，‘Iris-setosa’设为‘1’，‘Iris-versicolor’设为‘0’。
#测试数据集为训练数据集中随机抽取的十个数据，Iris-setosa和Iris-versicolor各五份
alpha = 0.8  
x = df[:,0:4]
x = x.T
y = df[:,4]
y = y.reshape((1,100))
w = nr.randn(4,1)
b = 0
c = np.array([1,2,3,4,5,6,7,8,9,10])
for i in range(1000):
    #正向传播
    Z = np.dot(w.T,x)+b
    A = 1/(1+np.exp(-1*Z))
    L = -(y*np.log(A)+(1-y)*np.log(1-A)) 
    #反向传播
    dz = A-y
    dw = np.dot(x,dz.T)/100
    db = (np.sum(dz))/100
    w = w-alpha*dw
    b = b-alpha*db
#读取测试数据
df = np.loadtxt(open("IrisText.txt","rb"),delimiter=",",skiprows=0) 
x = x = df[:,0:4]
x = x.T
y = df[:,4]
y = y.reshape((1,10))     
Z = np.dot(w.T,x)+b
A = 1/(1+np.exp(-1*Z))
print(A)
plt.scatter(c,A)
plt.scatter(c,y)
L = -(y*np.log(A)+(1-y)*np.log(1-A)) 
#本次实验主要用吴恩达老师所教的向量化方法进行逻辑回归练习，
#所用激活函数为sigmoid、函数，loss函数为交叉熵损失函数
#训练数据集为Iris数据集，后续应采用TensorFlow在复现一次
