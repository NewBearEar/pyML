import numpy as np
from geoidbn.rbm import RBM
from sklearn.neural_network import MLPRegressor
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict


class BPNet(nn.Module):
    def __init__(self,sizes,dropout):
        super(BPNet, self).__init__()
        self.sizes = sizes

        self.dropout = dropout
        n_input = self.sizes[0]
        n_output = self.sizes[-1]

        linearLayerDict = OrderedDict()
        # 构建网络层
        for k in range(len(self.sizes) - 1):
            lk = nn.Linear(self.sizes[k], self.sizes[k + 1])
            linearLayerDict.update({"l" + str(k): lk})
            if k != len(self.sizes) - 2:
                linearLayerDict.update({"sigmoid" + str(k): nn.LogSigmoid()})
        # 构建网络序列
        self.mlp = nn.Sequential(
            linearLayerDict
        )

    def forward(self, x):
        '''

        :param x:
        :return: OrderdDict，记录每个任务的输出
        '''
        y_pred = self.mlp(x)

        return y_pred