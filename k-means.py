# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 18:56:35 2019

@author: 木沐童
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 加载数据
def loadDataSet(start,end):
    #data = np.loadtxt(fileName,delimiter='\t')
    df = np.loadtxt(open("D:\CS\Graduate\暑假学习\算法\Data\wine.data","rb"),delimiter=",",skiprows=0) 
    data = df[:,start:end]
    
    return data
 
# 欧氏距离计算
def distEclud(x,y):
    return np.sqrt(np.sum((x-y)**2))  # 计算欧氏距离
 
# 为给定数据集构建一个包含K个随机质心的集合
def randCent(dataSet,k):
    m,n = dataSet.shape
    centroids = np.zeros((k,n))
    for i in range(k):
        index = int(np.random.uniform(0,m)) #
        centroids[i,:] = dataSet[index,:]
    return centroids
 
# k均值聚类
def KMeans(dataSet,k):
 
    m = np.shape(dataSet)[0]  #行的数目
    # 第一列存样本属于哪一簇
    # 第二列存样本的到簇的中心点的误差
    clusterAssment = np.mat(np.zeros((m,2)))
    clusterChange = True
 
    # 第1步 初始化centroids
    centroids = randCent(dataSet,k)
    while clusterChange:
        clusterChange = False
 
        # 遍历所有的样本（行数）
        for i in range(m):
            minDist = 100000.0
            minIndex = -1
 
            # 遍历所有的质心
            #第2步 找出最近的质心
            for j in range(k):
                # 计算该样本到质心的欧式距离
                distance = distEclud(centroids[j,:],dataSet[i,:])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            # 第 3 步：更新每一行样本所属的簇
            if clusterAssment[i,0] != minIndex:
                clusterChange = True
                clusterAssment[i,:] = minIndex,minDist**2
        #第 4 步：更新质心
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:,0].A == j)[0]]  # 获取簇类所有的点
            centroids[j,:] = np.mean(pointsInCluster,axis=0)   # 对矩阵的行求均值
 
    print("Congratulations,cluster complete!")
    return centroids,clusterAssment
 
def showCluster(dataSet,k,centroids,clusterAssment):
    m,n = dataSet.shape
    '''if n != 2:
        print("数据不是二维的")
        return 1'''
 
    mark = ['r', 'g', 'b', 'y', 'm', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("k值太大了")
        return 1
 
    # 绘制所有的样本
    for i in range(m):
        markIndex = int(clusterAssment[i,0])
        plt.scatter(dataSet[i,0],dataSet[i,1],c = mark[markIndex])
 
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # 绘制质心
    for i in range(k):
        plt.scatter(centroids[i,0],centroids[i,1],c='k')
 
    plt.show()
    
def showCluster3D(dataSet,k,centroids,clusterAssment):
    m,n = dataSet.shape
    ax = plt.figure().add_subplot(111, projection = '3d')
    mark = ['r','g','b','y','m'] 
    for i in range(m):
        markIndex = int(clusterAssment[i,0])
        x = dataSet[i,0]
        y = dataSet[i,1]
        z = dataSet[i,2]
        ax.scatter(x,y,z,c = mark[markIndex])
      
    for i in range(k):
        x = centroids[i,0]
        y = centroids[i,1]
        z = centroids[i,2]
        ax.scatter(x,y,z,c = 'k')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label') 
    plt.show()


dataSet = loadDataSet(1,4)
k = 3
centroids,clusterAssment = KMeans(dataSet,k)
 
showCluster3D(dataSet,k,centroids,clusterAssment)
'''--------------------- 
作者：寒夏12 
来源：CSDN 
原文：https://blog.csdn.net/hanxia159357/article/details/81530361 
版权声明：本文为博主原创文章，转载请附上博文链接！'''