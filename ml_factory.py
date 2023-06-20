from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

import xgboost as xgb
import lightgbm as lgbm
from neupy import utils
from neupy import algorithms
from geoidbn.dbn import DBN
from gtwgrnn.gtwgrnn import GTWGRNN


class ModelFactory():
    def __init__(self,model_name:str):
        '''
        模型工厂类，用于创建模型类实例对象
        :param model_name: 模型名称，需对应模型类名，暂时包括：RF，AdaBoost，GBDT，XGBoost，LightGBM，GRNN
                            LinearRegression,DBN
        '''
        # 输入的模型名称加上Model即为类名
        self.model_class_name = model_name + 'Model'
        self.__run_ml_module = __import__('run_ml')
        # 反射获取模型类
        self.__model_class = getattr(self.__run_ml_module,self.model_class_name)

    def getModelClassInstance(self):
        model = None
        if self.model_class_name == 'RFModel':
            # 调用sklearn随机森林构建模型,具体参数见官方文档
            model = RandomForestRegressor()
        elif self.model_class_name == 'AdaBoostModel':
            # 调用sklearn AdaBoost构建模型,具体参数见官方文档
            max_depth = 50
            model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=max_depth),
                                    learning_rate=0.8)
        elif self.model_class_name == 'GBDTModel':
            # 调用sklearn GBDT构建模型,具体参数见官方文档
            max_depth = 50
            n_estimators = 500
            model = GradientBoostingRegressor(loss='ls',n_estimators=n_estimators, max_depth=max_depth,
                                              learning_rate=0.1, subsample=0.8)
        elif self.model_class_name == 'XGBoostModel':
            # 调用xgboost包构建模型,具体参数见官方文档
            max_depth = 35
            n_estimators = 500
            model = xgb.XGBRegressor(max_depth=max_depth, learning_rate=0.09, n_estimators=n_estimators,
                                     objective='reg:gamma')
        elif self.model_class_name == 'LightGBMModel':
            # 调用lightgbm包构建模型,具体参数见官方文档
            max_depth = 9
            n_estimators = 2000
            model = lgbm.LGBMRegressor(boosting_type='gbdt', objective='regression', num_leaves=500,
                                    learning_rate=0.15, n_estimators=n_estimators, max_depth=max_depth,
                                    metric='rmse', bagging_fraction=0.8, feature_fraction=0.8, reg_lambda=0.5)
        elif self.model_class_name == 'GRNNModel':
            # 调用neupy包构建模型,具体参数见官方文档
            std = 0.037
            model = algorithms.GRNN(std=std)
        elif self.model_class_name == 'LinearRegressionModel':
            # 调用sklearn 线性回归构建模型,具体参数见官方文档
            model = LinearRegression()
        elif self.model_class_name == 'DBNModel':
            # DBN参数说明参考源码
            batch_size = 128
            model = DBN(neuralNum=[15, 15], learning_rate=0.0003, epochs=3000,
                               rbm_opts={"numepochs": 50,
                                         "batchsize": batch_size,
                                         "momentum": 0,
                                         "alpha": 1}
                               )
        elif self.model_class_name == 'GTWNNModel':
            # GTW GRNN模型 参考源码
            model = GTWGRNN(bandWidth=4,bLambda=3,spread=0.1)

        else:
            print('暂不支持该模型')
            return None
        self.__model_instance = self.__model_class(model)
        return self.__model_instance
