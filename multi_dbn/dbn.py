import numpy as np
from multi_dbn.rbm import RBM
from sklearn.neural_network import MLPRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from torch.autograd import Variable

class OPTS():
    def __init__(self,opt_dict=None):
        self.numepochs = 50
        self.batchsize = 128
        self.dbnBatchsize = 32
        self.momentum = 0
        self.alpha = 1
        if opt_dict:
            self.create_OPTS(opt_dict)
    def create_OPTS(self,opt_dict):
        try:
            self.numepochs = opt_dict["numepochs"]
            self.batchsize = opt_dict["batchsize"]
            self.dbnBatchsize = opt_dict["dbnBatchsize"]
            self.momentum = opt_dict["momentum"]
            self.alpha = opt_dict["alpha"]
        except Exception as e:
            print(e)
            print("opt dict 包含的不匹配")

class MultiTaskDBN():
    def __init__(self, neuralNum=[15, 15],targetNum = 2,weights=[0.5,0.5], learning_rate=0.001, epochs=1000,use_gpu=False, rbm_opts=None):
        '''

        :param neuralNum:
        :param targetNum:指定多任务的target维度
        :param weights: 指定多任务的权重，list，维度与targetNum一致
        :param learning_rate: 学习率
        :param epochs: epoch迭代数量
        :param rbm_opts: rbm参数字典，至少包括，numepochs迭代次数,batchsize指rbm的batch,dbnBatchsize指dbn网络的batch,momentum动量值,alpha学习率
        '''
        self.targetNum = targetNum
        self.weights = weights
        # 隐藏层单元个数 如[15 15]
        self.sizes = neuralNum
        # 训练的模型
        self.net = None
        # 存储rbm参数
        self.opts = OPTS(rbm_opts)
        # mlp学习率
        self.lr = learning_rate
        # epoch
        self.epochs = epochs
        # 防止重复建立rbm
        self.hasRBM = False
        self.gpu = use_gpu
        if torch.cuda.is_available() and use_gpu:
            self.device = torch.device('cuda')
            self.gpu = True
        else:
            self.device = torch.device('cpu')
            self.gpu=False

    def __dbnSetup(self,x,y,rbm_opts):
        if self.hasRBM == False:
            n_input = x.shape[1]
            n_output = y.shape[1]
            self.sizes = np.concatenate(([n_input],self.sizes,[n_output]))
            self.rbm_list = []
            for u in range(0,len(self.sizes)-1):
                rbm = RBM(self.sizes[u],self.sizes[u+1])
                rbm.rbmsetup(rbm_opts)
                self.rbm_list.append(rbm)
            self.hasRBM = True

    def __dbnPreTrain(self,x,rbm_opts):
        if self.hasRBM == True:
            n = len(self.rbm_list)
            self.rbm_list[0].rbmtrain(x,rbm_opts)
            for i in range(1,n):
                x = self.rbm_list[i-1].rbmup(x)
                self.rbm_list[i].rbmtrain(x,rbm_opts)
        else:
            print("rbm预训练之前必须创建rbm")
    def __gradient_descent(self,y_pred,y,optimizer,loss_fn,loss_weights):
        loss = 0
        # 先默认只有三个任务，跑通再重构
        for k in range(y.shape[1]):
            perloss = (loss_fn(y_pred[k],y[:,k].reshape(-1,1)))/y.shape[0] * loss_weights[k]
            loss += perloss
        #loss1 = loss_fn(y_pred[0],y[:,0].reshape(-1,1))
        #loss2 = loss_fn(y_pred[1],y[:,1].reshape(-1,1))
        #loss3 = loss_fn(y_pred[2],y[:,2].reshape(-1,1))
        #loss = loss1+loss2+loss3
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def __gradient_descent_uncertainty(self,y_pred,y,optimizer,lvars):
        loss = 0
        # 先默认只有三个任务，跑通再重构

        for k in range(y.shape[1]):
            #precision = 1/2*(torch.exp(log_vars[k])**2)
            #precision = torch.exp(-log_vars[k])
            precision = 1/(2*(lvars[k]**2))
            diff = (y_pred[k]-y[:,k].reshape(-1,1))**2
            loss += torch.sum(10*precision*diff + torch.log(lvars[k]) ,-1)
        loss = torch.mean(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def fit(self,input,target,loss_weights=[],dropout=0):
        '''

        :param uncertainty:
        :param auto_lr:
        :param input: 输入变量
        :param target: 目标值
        :param loss_weights: 多任务权重 , 权重维度与target的维度相同 ,可以重新给定权重
        :param plot_acc: 是否绘制模型训练loss和R2曲线
        :param show_train_process: 显示模型训练过程
        :return:
        '''

        if (target.shape[1] != self.targetNum):
            self.targetNum = target.shape[1]

        if loss_weights:
            loss_weights = loss_weights
        else:
            loss_weights = self.weights

        assert self.targetNum == len(loss_weights)

        show_train_process = True
        auto_lr = False
        uncertainty = False

        torch.set_printoptions(precision=8)
        if self.device == torch.device('cuda'):
            print("使用GPU训练")
        else:
            print("使用CPU训练")

        opts = self.opts
        #print(opts)
        self.__dbnSetup(input,target,opts)
        self.__dbnPreTrain(input,opts)

        x = Variable(torch.tensor(input).to(torch.float32)).to(self.device)
        y = Variable(torch.tensor(target).to(torch.float32)).to(self.device)
        # 数据加入gpu
        #if(self.gpu):
        #    x,y = x.cuda(),y.cuda()
        # 构建网络
        self.net = BPNet(self.sizes,self.rbm_list,dropout).to(self.device)
        # 定义损失函数
        loss_fn = nn.MSELoss(reduction='mean').to(self.device)
        # 模型加入gpu
        #if(self.gpu):
        #    self.net.cuda()
        #    loss_fn.cuda()
        learning_rate = self.lr
        epochs = self.epochs
        params = self.net.parameters()
        # 不确定性系数
        #log_var_a = torch.zeros((1,), requires_grad=True)
        #log_var_b = torch.zeros((1,), requires_grad=True)
        lvar_a = torch.ones((1,),requires_grad=True)
        lvar_b = torch.ones((1,), requires_grad=True)
        if uncertainty:
            # 构建同方差不确定性
            params = ([p for p in params]+[lvar_a]+[lvar_b])
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        scheduler = None
        # 学习率自动下降
        if auto_lr:
            lr_lambda = lambda step: (1.0 - step / epochs) if step <= epochs else 0
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lr_lambda,last_epoch=-1)
        # 样本数
        m = x.shape[0]
        numbatches = int(m // opts.dbnBatchsize)
        batchsize = opts.dbnBatchsize
        for t in range(epochs):
            kk = np.arange(m)
            np.random.shuffle(kk)
            cost = 0
            # 记录真实batch数
            n_batch = 0
            for b in range(1, numbatches + 1):
                # y_pred = self.net(x)
                x_batch = x[kk[(b - 1) * batchsize:b * batchsize], :]
                y_batch = y[kk[(b - 1) * batchsize:b * batchsize], :]
                # 正向传播
                y_pred_batch = self.net(x_batch)
                # 计算损失，梯度下降
                if uncertainty:
                    loss = self.__gradient_descent_uncertainty(y_pred_batch, y_batch,optimizer, [lvar_a,lvar_b])
                else:
                    loss = self.__gradient_descent(y_pred_batch, y_batch, optimizer, loss_fn, loss_weights=loss_weights)
                cost += loss
                n_batch += 1
            if m % batchsize != 0:
                # 获取最后剩余的部分
                x_batch_end = x[kk[numbatches * batchsize:], :]
                y_batch_end = y[kk[numbatches * batchsize:], :]
                y_pred_batch = self.net(x_batch_end)
                # 计算损失，梯度下降
                if uncertainty:
                    loss = self.__gradient_descent_uncertainty(y_pred_batch, y_batch_end,optimizer, [lvar_a,lvar_b])
                else:
                    loss = self.__gradient_descent(y_pred_batch, y_batch_end, optimizer, loss_fn,loss_weights)
                cost += loss
                n_batch += 1
            if auto_lr:
                scheduler.step()
            if show_train_process:
                if t % 200 == 0:
                    mean_cost = cost / n_batch
                    if self.gpu:
                        mean_cost = mean_cost.cpu()
                    print("epoch {} cost:".format(t))
                    print(mean_cost.data.detach().numpy())
                    print("Lr:{}".format(optimizer.state_dict()['param_groups'][0]['lr']))
                    if uncertainty:
                        #precision_a = torch.exp(-log_var_a)
                        #precision_b = torch.exp(-log_var_b)
                        precision_a=1 / (2 * (lvar_a ** 2))
                        precision_b=1 / (2 * (lvar_b ** 2))
                        print("auto_weight_a:{}".format(precision_a))
                        print("auto_weight_b:{}".format(precision_b))
        if uncertainty:
            #precision_a = torch.exp(-log_var_a)
            #precision_b = torch.exp(-log_var_b)
            precision_a = 1 / (2 * (lvar_a ** 2))
            precision_b = 1 / (2 * (lvar_b ** 2))
            print("final:")
            print("auto_weight_a:{}".format(precision_a))
            print("auto_weight_b:{}".format(precision_b))
            return precision_a,precision_b
        return
    def predict(self, x):
        '''

        :param x: numpy形式的输入值
        :return: numpy形式的输出值
        '''
        x_pred = Variable(torch.tensor(x).to(torch.float32)).to(self.device)

        y_pred = self.net(x_pred)
        num = 0
        if self.gpu:
            for yp in y_pred:
                y_pred[num] = yp.cpu()
                num += 1
        return y_pred

class BPNet(nn.Module):
    def __init__(self,sizes,rbm_list,dropout):
        super(BPNet, self).__init__()
        self.sizes = sizes
        self.rbm_list = rbm_list
        self.dropout = dropout
        n_input = self.sizes[0]
        n_output = self.sizes[-1]

        linearLayerDict = OrderedDict()
        # 构建网络层,共享部分
        # 两层private层结构
        for k in range(len(self.sizes) - 3):
            lk = nn.Linear(self.sizes[k], self.sizes[k + 1])
            lk.weight = torch.nn.Parameter(torch.Tensor(self.rbm_list[k].W))
            lk.bias = torch.nn.Parameter(torch.Tensor(self.rbm_list[k].c).squeeze())
            linearLayerDict.update({"l" + str(k): lk})
            if k != len(self.sizes) - 2:
                if dropout != 0:
                    linearLayerDict.update({"dropout"+str(k):nn.Dropout(self.dropout)})
                linearLayerDict.update({"sigmoid" + str(k): nn.LogSigmoid()})
        self.mlp_shared_layers = nn.Sequential(
            linearLayerDict
        )
        #self.l1 = nn.Linear(15,1)
        # 多任务输出部分
        self.allParamDict = self.__dict__#OrderedDict()
        tempNames = locals()
        for k in range(n_output):
            lk = nn.Linear(self.sizes[-3],self.sizes[-2])
            lk.weight = torch.nn.Parameter(torch.Tensor(self.rbm_list[-2].W))
            lk.bias = torch.nn.Parameter(torch.Tensor(self.rbm_list[-2].c).squeeze())
            outlk = nn.Linear(self.sizes[-2],1)
            outlk.weight = torch.nn.Parameter(torch.Tensor(np.reshape(self.rbm_list[-1].W[k,:],(1,-1))))
            outlk.bias = torch.nn.Parameter(torch.Tensor(self.rbm_list[-1].c[k,:]).squeeze())

            tempNames["output_l"+str(k)] = nn.Sequential(OrderedDict({
                "l"+str(k):lk,
                "dropout_l"+str(k):nn.Dropout(self.dropout),
                "sigmoid_l"+str(k):nn.LogSigmoid(),
                "outl"+str(k):outlk
            }))
            self._modules.update({"output_l"+str(k): tempNames["output_l"+str(k)]})

    def forward(self, x):
        '''

        :param x:
        :return: OrderdDict，记录每个任务的输出
        '''
        out_shared = self.mlp_shared_layers(x)
        outList = []
        for k in range(self.sizes[-1]):
            output_lk = self._modules["output_l"+str(k)]
            #print(output_lk(out_shared))
            outList.append(output_lk(out_shared))
        return outList

class MultiLoss(nn.Module):
    def __init__(self):
        super(MultiLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='sum')

    def forward(self, y_pred, y_real):
        task_loss_1 = self.loss_fn(y_pred[:,0],y_real[:,0])
        task_loss_2 = self.loss_fn(y_pred[:,1],y_real[:,1])
        task_loss_3 = self.loss_fn(y_pred[:,2],y_real[:,2])
        print(y_real[:,0])
        #multi_loss = (task_loss_1 + task_loss_2 + task_loss_3)/3.0
        multi_loss = task_loss_1
        return multi_loss



