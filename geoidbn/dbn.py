import numpy as np
from geoidbn.rbm import RBM
from sklearn.neural_network import MLPRegressor
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict

class OPTS():
    def __init__(self,opt_dict=None):
        self.numepochs = 50
        self.batchsize = 128
        self.momentum = 0
        self.alpha = 1
        if opt_dict:
            self.create_OPTS(opt_dict)
    def create_OPTS(self,opt_dict):
        try:
            self.numepochs = opt_dict["numepochs"]
            self.batchsize = opt_dict["batchsize"]
            self.momentum = opt_dict["momentum"]
            self.alpha = opt_dict["alpha"]
        except Exception as e:
            print(e)
            print("opt dict 包含的不匹配")


class DBN():
    def __init__(self,neuralNum=[15,15],learning_rate = 0.001,epochs = 1000,rbm_opts = None):
        # 隐藏层单元个数 如[15 15]
        self.sizes = neuralNum
        # 训练的模型
        self.mlp = None
        # 存储rbm参数
        self.opts = OPTS(rbm_opts)
        # mlp学习率
        self.lr = learning_rate
        # epoch
        self.epochs = epochs

    def __dbnSetup(self,x,y,rbm_opts):
        n_input = x.shape[1]
        n_output = y.shape[1]
        self.sizes = np.concatenate(([n_input],self.sizes,[n_output]))
        self.rbm_list = []
        for u in range(0,len(self.sizes)-1):
            rbm = RBM(self.sizes[u],self.sizes[u+1])
            rbm.rbmsetup(rbm_opts)
            self.rbm_list.append(rbm)

    def __dbnPreTrain(self,x,rbm_opts):
        n = len(self.rbm_list)
        self.rbm_list[0].rbmtrain(x,rbm_opts)
        for i in range(1,n):
            x = self.rbm_list[i-1].rbmup(x)
            self.rbm_list[i].rbmtrain(x,rbm_opts)

    def __gradient_descent(self,y_pred,y,optimizer,loss_fn):
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def fit(self,input,target):
        '''

        :param input: 输入变量
        :param target: 目标值
        :param learning_rate: 学习率
        :param epochs: epoch迭代数量
        :param rbm_opts: rbm参数字典，至少包括，numepochs迭代次数,batchsize,momentum动量值,alpha学习率
        :return:
        '''
        opts = self.opts
        #print(opts)
        self.__dbnSetup(input,target,opts)
        self.__dbnPreTrain(input,opts)

        x = Variable(torch.tensor(input).to(torch.float32))
        y = Variable(torch.tensor(target).to(torch.float32))
        linearLayerDict = OrderedDict()
        # 构建网络层
        for k in range(len(self.sizes)-1):
            lk = nn.Linear(self.sizes[k], self.sizes[k+1])
            lk.weight = torch.nn.Parameter(torch.Tensor(self.rbm_list[k].W))
            lk.bias = torch.nn.Parameter(torch.Tensor(self.rbm_list[k].c).squeeze())
            linearLayerDict.update({"l"+str(k) : lk})
            if k != len(self.sizes)-2:
                linearLayerDict.update({"sigmoid"+str(k) : nn.LogSigmoid()})

        for rbm in self.rbm_list:
            print(rbm.W.shape,rbm.c.shape)
        # 构建网络序列
        self.mlp = nn.Sequential(
            linearLayerDict
        )
        print(self.mlp.parameters())
        # 定义损失函数
        loss_fn = nn.MSELoss(reduction='mean')
        learning_rate = self.lr
        epochs = self.epochs
        optimizer = torch.optim.Adam(self.mlp.parameters(),lr=learning_rate)
        print(self.mlp)
        self.mlp.parameters()
        # 样本数
        m = x.shape[0]
        numbatches = int(m // opts.batchsize)
        batchsize = opts.batchsize
        for t in range(epochs):
            kk = np.arange(m)
            np.random.shuffle(kk)
            cost = 0
            # 记录真实batch数
            n_batch = 0
            for b in range(1,numbatches+1):
                #y_pred = self.mlp(x)
                x_batch = x[kk[(b-1)*batchsize:b*batchsize],:]
                y_batch = y[kk[(b-1)*batchsize:b*batchsize],:]
                # 正向传播
                y_pred_batch = self.mlp(x_batch)
                # 计算损失，梯度下降
                loss = self.__gradient_descent(y_pred_batch,y_batch,optimizer,loss_fn)
                cost += loss
                n_batch += 1
            if m % opts.batchsize != 0:
                # 获取最后剩余的部分
                x_batch_end = x[kk[numbatches*batchsize:],:]
                y_batch_end = y[kk[numbatches*batchsize:],:]
                y_pred_batch = self.mlp(x_batch_end)
                # 计算损失，梯度下降
                loss = self.__gradient_descent(y_pred_batch, y_batch_end, optimizer, loss_fn)
                cost += loss
                n_batch += 1

            mean_cost = cost/n_batch
            print("epoch "+str(t)+" cost:")
            print(mean_cost.data.detach().numpy())
    def predict(self,x):
        '''

        :param x: numpy形式的输入值
        :return: numpy形式的输出值
        '''
        y_pred = self.mlp(Variable(torch.tensor(x).to(torch.float32)))
        return y_pred.detach().numpy()


class BPNet(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(BPNet, self).__init__()




