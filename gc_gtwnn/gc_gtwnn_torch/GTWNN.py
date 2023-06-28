import numpy as np
from geoidbn.rbm import RBM
from sklearn.neural_network import MLPRegressor
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
from gc_gtwnn import BasicFunc
import copy
class GCGTWNN():
    def __init__(self,bandWidth=4,bLambda=3,neuralNum=[15, 15],learning_rate=0.001, gc_epochs=1000,ft_epochs=1000,dropout=0):
        '''

        :param bandWidth:
        :param bLambda:
        :param neuralNum:
        :param learning_rate:
        :param gc_epochs: 全局 epoch
        :param ft_epochs: 微调 epoch
        :param dropout:
        '''
        self.bandWidth = bandWidth
        self.bLambda = bLambda * 10e4
        self.sizes = neuralNum
        self.X_train = None
        self.y_train = None
        self.train_coor = None
        self.train_t = None
        self.net = None
        self.lr = learning_rate
        self.gc_epochs = gc_epochs
        self.ft_epochs = ft_epochs
        self.dropout = dropout

    def fit(self,X_train,y_train):
        """
            Parameters
            ----------
            X_train : array-like (n_samples, n_features)
            但必须保证前三个维度为 时间（day of year), 纬度 ，经度

            y_train : array-like (n_samples,)
            ------
        """
        train_coor = X_train[:,1:3]
        train_t = X_train[:,0].reshape(-1,1)
        self.fit_gc(train_coor,train_t,X_train[:,3:],y_train)


    def fit_gc(self,train_coor,train_t,X_train,y_train):
        self.train_coor = train_coor
        self.train_t = train_t.reshape(-1,1)
        self.X_train = X_train
        self.y_train =y_train
        dropout = self.dropout
        # 全局训练
        X = Variable(torch.tensor(X_train).to(torch.float32))
        y = Variable(torch.tensor(y_train.reshape(-1,1)).to(torch.float32))

        n_input = X.shape[1]
        n_output = y.shape[1]
        self.sizes = np.concatenate(([n_input], self.sizes, [n_output]))

        self.net = BPNet(self.sizes, dropout)
        loss_fn = nn.MSELoss(reduction='mean')
        params = self.net.parameters()
        optimizer = torch.optim.Adam(params, lr=self.lr)

        epochs = self.gc_epochs
        for t in range(epochs):
            y_pred = self.net(X)
            loss = self.__gradient_descent(y_pred, y, optimizer, loss_fn)

            print("epoch " + str(t) + " loss:")
            print(loss.data.detach().numpy())

    def finetune(self,ft_coor,ft_t,X_pred,y_pred):
        print("推理前微调...")

        if not self.net:
            print("must fit_gc first")
            return
        ft_t = ft_t.reshape(-1,1)
        self.finetune_net_list = []
        val_allestimated = []
        val_allobserved = []
        val_alltime = []
        val_alllat = []
        val_alllon = []
        for i in range(0, len(y_pred)):  # this: val_time[i], val_lat[i], val_lon[i]
            # adaptive bandwidth
            bd = BasicFunc.calc_bandwidth_via_points(ft_t[i,0].reshape(-1,1), ft_coor[i,0].reshape(-1,1),ft_coor[i,1].reshape(-1,1), self.train_t.reshape(-1,1), self.train_coor[:,0].reshape(-1,1),
                                                     self.train_coor[:,1].reshape(-1,1), self.bandWidth)
            # calculating the spatiotemporal weighting
            thisw = BasicFunc.calc_wMatrix(ft_t[i,0].reshape(-1,1), ft_coor[i,0].reshape(-1,1),ft_coor[i,1].reshape(-1,1), self.train_t.reshape(-1,1), self.train_coor[:,0].reshape(-1,1),
                                                     self.train_coor[:,1].reshape(-1,1), bd, self.bLambda)
            included = thisw > 0.000001
            included = included.flatten()
            valid_train_w = thisw[included]
            valid_train_w = np.around(valid_train_w, decimals=6)
            valid_train_x = self.X_train[included,:]
            valid_train_y = self.y_train[included]

            this_train_x = valid_train_x
            this_val_x = X_pred[i, :]
            this_val_x.shape = 1, len(this_val_x)
            this_val_x_T = Variable(torch.tensor(this_val_x).to(torch.float32))
            # 对于每个位置拷贝一个新的网络
            finetune_net = copy.deepcopy(self.net)


            if this_train_x.shape[0] < 1:  # <6 samples are collected
                val_y_pred = finetune_net(this_val_x)
                val_allestimated.append(val_y_pred.detach().numpy())
                val_allobserved.append(y_pred[i])
                val_alltime.append(ft_t[i,0])
                val_alllat.append(ft_coor[i,0])
                val_alllon.append(ft_coor[i,1])
                print('sample<2, non-fine-tuning')
                continue
            # 开始微调
            loss_fn = GTWLoss()
            params = self.net.parameters()
            optimizer = torch.optim.Adam(params, lr=self.lr)

            epochs = self.ft_epochs
            for t in range(epochs):
                X = Variable(torch.tensor(valid_train_x).to(torch.float32))
                y = Variable(torch.tensor(valid_train_y.reshape(-1, 1)).to(torch.float32))
                valid_train_y_pred = finetune_net(X)
                loss = self.__gradient_descent_gtw(valid_train_y_pred, y, optimizer, loss_fn,geoW=Variable(torch.tensor(valid_train_w)))
                loss = loss/X.shape[0]
                print("epoch " + str(t) + " loss:")
                print(loss.data.detach().numpy())

            self.finetune_net_list.append(finetune_net)
            # 预测当前点
            val_y_pred = finetune_net(this_val_x_T).item()
            val_allestimated.append(val_y_pred)
            val_allobserved.append(y_pred[i])
            val_alltime.append(ft_t[i, 0])
            val_alllat.append(ft_coor[i, 0])
            val_alllon.append(ft_coor[i, 1])
        return val_allestimated,val_allobserved,val_alltime,val_alllat,val_alllon
    def predict(self,X_pred,y_pred_real=None):
        """
                    Parameters
                    ----------
                    X_predict : array-like (n_samples, n_features)
                    待预测数据的自变量值
                    但必须保证前三个维度为 时间（day of year), 纬度 ，经度

                    y_real : array-like (n_samples,)
                    待遇测数据的因变量 真实值。如果没有真实值标签（在推理时）输入None
                    ------
                """
        self.timeLoc_predict = X_pred[:, 0:3]
        ft_coor = X_pred[:, 1:3]
        ft_t = X_pred[:, 0].reshape(-1,1)

        if y_pred_real:
            self.y_predict_real = y_pred_real
        else:
            y_pred_real = np.zeros(X_pred.shape[0])
            self.y_predict_real = y_pred_real

        # 包含微调过程
        y_pred_predict,_,_,_,_ = self.finetune(ft_coor,ft_t,X_pred[:, 3:],y_pred_real.reshape(-1,1))
        return np.array(y_pred_predict)
    def __gradient_descent(self,y_pred,y,optimizer,loss_fn):
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def __gradient_descent_gtw(self,y_pred,y,optimizer,loss_fn,geoW):
        loss = loss_fn(y_pred, y,geoW)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

class GTWLoss(nn.Module):
    def __init__(self):
        super(GTWLoss, self).__init__()

    def forward(self, y_pred, y_real,geoW):
        return 0.5 * torch.mean(geoW * torch.pow((y_real - y_pred), 2))

class BPNet(nn.Module):
    def __init__(self,sizes,dropout):
        super(BPNet, self).__init__()
        self.sizes = sizes

        self.dropout = dropout

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