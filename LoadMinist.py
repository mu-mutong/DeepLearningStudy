# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 18:10:00 2019

@author: 木沐童
"""

train_image_path = 'train-images.idx3-ubyte'
train_label_path = 'train-labels.idx1-ubyte'
test_image_path = 't10k-images.idx3-ubyte'
test_label_path = 't10k-labels.idx1-ubyte'

import numpy as np
from struct import unpack
from PIL import Image


def __read_image(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img = np.fromfile(f, dtype=np.uint8).reshape(num, 784)
    return img

def __read_label(path):
    with open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        lab = np.fromfile(f, dtype=np.uint8)
    return lab
    
def __normalize_image(image):
    img = image.astype(np.float32) / 255.0
    return img

def __one_hot_label(label):
    lab = np.zeros((label.size, 10))
    for i, row in enumerate(lab):
        row[label[i]] = 1
    return lab

def load_mnist(train_image_path, train_label_path, test_image_path, test_label_path, normalize=True, one_hot=True):
    '''读入MNIST数据集
    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label : 
        one_hot为True的情况下，标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    Returns
    ----------
    (训练图像, 训练标签), (测试图像, 测试标签)
    '''
    image = {
        'train' : __read_image(train_image_path),
        'test'  : __read_image(test_image_path)
    }

    label = {
        'train' : __read_label(train_label_path),
        'test'  : __read_label(test_label_path)
    }
    
    if normalize:
        for key in ('train', 'test'):
            image[key] = __normalize_image(image[key])

    if one_hot:
        for key in ('train', 'test'):
            label[key] = __one_hot_label(label[key])

    return (image['train'], label['train']), (image['test'], label['test'])
def Draw(train,n):
    I = train[n]
    I.resize((28, 28))
    im = Image.fromarray((I*256).astype('uint8'))
    im.show()

if __name__ == '__main__':
    (x,y),(tx,ty) = load_mnist(train_image_path, train_label_path, test_image_path, test_label_path, normalize=True, one_hot=True)
   
    
    
'''--------------------- 
作者：孤独暗星 
来源：CSDN 
原文：https://blog.csdn.net/weixin_40522523/article/details/82823812 
版权声明：本文为博主原创文章，转载请附上博文链接！'''
