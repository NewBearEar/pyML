import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from abc import ABCMeta
import time

import xgboost as xgb
import lightgbm as lgbm
from neupy import utils
from neupy import algorithms

from scoring_method import cal_Rsqure,cal_MAE,cal_RMSE,cal_RPE
#from dbn import SupervisedDBNRegression
from geoidbn.dbn import DBN

class MLModel(metaclass=ABCMeta):
    def __init__(self):
        '''
        # 初始化
        :param dataset: 包含数据的DataFrame,列为特征,行为样本
        '''
        self.dataset = None
        self.scaler = MinMaxScaler()
        self.model = None
    def load_data(self,dataset):
        self.dataset = dataset
        self.__data_prepare()

    def __data_prepare(self):
        # ipnut : AOD;RH;WS;TEMP;PBL;SP;NDVI;PMs;PMt;1./NearestDis
        # output : PM2.5
        # 最后一列默认是输出变量
        self.input = self.dataset.iloc[:, :-1]
        self.output = self.dataset.iloc[:, -1]
        # 归一化
        self.scaled_input = self.scaler.fit_transform(np.array(self.input))
        # 输出不需要归一化
        self.scaled_output = np.array(self.output).reshape((-1,1))

    def __train_model(self,input_data,output_data,mode:str):
        '''

        :param input_data: 训练集输入变量
        :param output_data: 训练集输出变量
        :param mode: 不同库有不同接口,选择训练模型调用的函数，train or fit
        :return:
        '''
        if 'fit' == mode:
            self.model.fit(input_data,output_data)
        elif 'train' == mode:
            self.model.train(input_data,output_data)
        else:
            raise RuntimeError('未定义的训练方法')

    def __tenFolder(self,mode):
        '''

        :param mode: 不同库有不同接口,选择训练模型调用的函数，train or fit
        :return:
        '''
        # 随机数种子
        rs = 20
        # 十折分割
        kf = KFold(n_splits=10, shuffle=True, random_state=20)
        best_model = self.model
        best_R2_val = 0.0
        best_train_target = None
        best_train_estimated = None
        # 真值数组
        val_all = np.array(0)
        # 估计值数组
        val_estimated_all = np.array(0)
        count = 1
        for train_idx, test_idx in kf.split(self.scaled_input):
            self.__train_model(self.scaled_input[train_idx, :], self.scaled_output[train_idx], mode)

            scaled_train_estimated = self.model.predict(self.scaled_input[train_idx, :])
            scaled_val_estimated = self.model.predict(self.scaled_input[test_idx, :])
            # 反归一化
            train_target = self.scaler.inverse_transform(self.scaled_output[train_idx])
            train_estimated = self.scaler.inverse_transform(scaled_train_estimated.reshape(-1, 1))
            val_estimated = self.scaler.inverse_transform(scaled_val_estimated.reshape(-1, 1))
            # 将验证数据加入数组
            val_true = np.array(self.output)[test_idx]

            val_all = np.vstack((val_all, val_true.reshape((-1, 1))))
            val_estimated_all = np.vstack((val_estimated_all, val_estimated.reshape((-1, 1))))

            # 计算R2
            R2_temp = cal_Rsqure(val_true, val_estimated)
            print('第', count, '折交叉验证循环结束')
            count += 1
            if R2_temp > best_R2_val:
                best_R2_val = R2_temp
                best_model = self.model
                best_train_target = train_target
                best_train_estimated = train_estimated
        self.best_R2_val = best_R2_val
        self.best_model = best_model
        self.best_train_target = best_train_target
        self.best_train_estimated = best_train_estimated
        self.val_all = val_all
        self.val_estimated_all = val_estimated_all

    def model_fitting(self):
        '''

        :return: 训练集输出真值，估计值，训练后的模型
        '''
        time_start = time.time()
        self.__train_model(self.scaled_input, self.scaled_output,'fit')
        scaled_train_estimated = self.model.predict(self.scaled_input)
        # 反归一化
        self.train_estimated = self.scaler.inverse_transform(scaled_train_estimated.reshape(-1, 1))
        time_end = time.time()
        print('model fiiting time cost:', time_end - time_start, 's')
        return np.array(self.output), self.train_estimated, self.model

    def model_tenFolder(self):
        '''

        :return: 十折打乱后的全部真值，全部估计值，最优模型，最优一折的模型训练集，最优一折的训练集估计值
        '''
        time_start = time.time()
        self.__tenFolder('fit')
        time_end = time.time()
        print('ten folder time cost:', time_end - time_start, 's')
        print("best model R2 : ", self.best_R2_val)
        return self.val_all,self.val_estimated_all,self.best_model,self.best_train_target,self.best_train_estimated

class RFModel(MLModel):
    def __init__(self,model:RandomForestRegressor):
        '''
        封装后的随机森林模型类
        :param dataset:
        :param model: 实际的模型对象
        '''
        super(RFModel, self).__init__()
        self.model = model

class AdaBoostModel(MLModel):
    def __init__(self,model:AdaBoostRegressor):
        super(AdaBoostModel, self).__init__()
        self.model = model

class GBDTModel(MLModel):
    def __init__(self,model:GradientBoostingRegressor):
        super(GBDTModel, self).__init__()
        self.model = model

class XGBoostModel(MLModel):
    def __init__(self,model:xgb.XGBRegressor):
        super(XGBoostModel, self).__init__()
        self.model = model

class LightGBMModel(MLModel):
    def __init__(self,model:lgbm.LGBMRegressor):
        super(LightGBMModel, self).__init__()
        self.model = model

class DBNModel(MLModel):
    def __init__(self,model:DBN):
        super(DBNModel, self).__init__()
        self.model = model

class GRNNModel(MLModel):
    def __init__(self,model:algorithms.GRNN):
        super(GRNNModel, self).__init__()
        self.model = model

    def model_fitting(self):    # 覆写父类方法
        time_start = time.time()
        self.__train_model(self.scaled_input, self.scaled_output,'train')
        scaled_train_estimated = self.model.predict(self.scaled_input)
        # 反归一化
        #self.train_estimated = self.scaler.inverse_transform(scaled_train_estimated.reshape(-1, 1))
        time_end = time.time()
        print('time cost:', time_end - time_start, 's')
        return np.array(self.output), self.train_estimated, self.model

    def model__tenFolder(self):    # 覆写父类方法
        time_start = time.time()
        self.__tenFolder('train')
        time_end = time.time()
        print('time cost:', time_end - time_start, 's')
        print("best model R2 : ", self.best_R2_val)
        return self.val_all, self.val_estimated_all, self.best_model, self.best_train_target, self.best_train_estimated

class LinearRegressionModel(MLModel):
    def __init__(self,model:LinearRegression):
        super(LinearRegressionModel, self).__init__()
        self.model = model

