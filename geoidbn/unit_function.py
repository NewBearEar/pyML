import numpy as np

def sigmoid(inx):
    #对sigmoid函数的优化，避免了出现极大的数据溢出
    index1 = (inx >= 0)
    index2 = (inx < 0)

    z = np.zeros(inx.shape)
    z[index1] = 1.0/(1+np.exp(-inx[index1]))
    z[index2] = np.exp(inx[index2])/(1+np.exp(inx[index2]))
    return z

def sigm(x):
    z = sigmoid(x)
    return z

def sigmrnd(x):
    Z = sigmoid(x) > np.random.rand(*x.shape)
    return Z
