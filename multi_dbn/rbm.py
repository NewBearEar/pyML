import numpy as np
from multi_dbn.unit_function import sigm,sigmrnd

class RBM():
    def __init__(self,v_size,h_size):
        # 显层size
        self.v_size = v_size
        self.h_size = h_size

    def rbmsetup(self,opts):
        self.alpha = opts.alpha
        self.momentum = opts.momentum

        self.W = 0.1 * np.random.randn(self.h_size,self.v_size)
        self.vW = np.zeros((self.h_size,self.v_size))

        self.b = np.zeros((self.v_size,1))
        self.vb = np.zeros((self.v_size,1))

        self.c = np.zeros((self.h_size,1))
        self.vc = np.zeros((self.h_size,1))

    def rbmtrain(self,x,opts):
        # 样本数m
        m = x.shape[0]
        numbatches = int(m//opts.batchsize)
        for i in range(1,opts.numepochs+1):
            kk = np.arange(m)
            np.random.shuffle(kk)
            err = 0
            for j in range(1,numbatches+1):

                batch = x[kk[(j-1)* opts.batchsize:j*opts.batchsize],:]
                #print(batch.shape)

                v1 = batch
                h1 = sigmrnd(self.c.T + np.dot(v1, self.W.T))
                v2 = sigmrnd(self.b.T + np.dot(h1,self.W))
                h2 = sigm(self.c.T + np.dot(v2,self.W.T))

                c1 = np.dot(h1.T , v1)
                c2 = np.dot(h2.T , v2)
                # 动量梯度下降
                self.vW = self.momentum * self.vW + self.alpha * (c1-c2) / opts.batchsize
                self.vb = self.momentum * self.vb + self.alpha * np.sum(v1-v2).T / opts.batchsize
                self.vc = self.momentum * self.vc + self.alpha * np.sum(h1-h2).T / opts.batchsize

                self.W += self.vW
                self.b += self.vb
                self.c += self.vc

                err = err + np.sum(np.sum(np.power(v1-v2,2)))/opts.batchsize
        print('one RBM training completed.')

    def rbmup(self,x):
        z = sigm(self.c.T+np.dot(x,self.W.T))
        return z


