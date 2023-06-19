
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
#import lightgbm as lgbm
from neupy import utils
from neupy import algorithms
import pandas as pd
import numpy as np
import time

from ml_factory import ModelFactory
from scoring_method import cal_Rsqure,cal_MAE,cal_RMSE,cal_RPE


from warnings import simplefilter,filterwarnings
simplefilter(action="ignore",category=FutureWarning)
filterwarnings("ignore")
'''
ML_toolbox_v1.0 
使用demo
'''
scaler = StandardScaler()#MinMaxScaler()
dataset = pd.read_csv(
        r'.\testData\sample_data_noST.csv')
input = dataset.iloc[:, :-1]
output = dataset.iloc[:, -1]
# 归一化
scaled_input = scaler.fit_transform(np.array(input))
scaled_output = scaler.fit_transform(np.array(output).reshape((-1,1)))


train_X_PMst,test_X_PMst, train_Y_PMst, test_Y_PMst = train_test_split(scaled_input,
                                                   scaled_output,
                                                   test_size = 0.1,
                                                   random_state = 2)

'''
获取模型工厂实例，以DBN为例
模型名称，需对应模型类名，暂时包括：RF，AdaBoost，GBDT，XGBoost，LightGBM，GRNN
                            LinearRegression,DBN
'''
factory = ModelFactory("DBN")
# 获取模型类
modelClass = factory.getModelClassInstance()

time_start = time.time()
# 模型训练 fit 函数或者 train函数
print("Start Training")
modelClass.model.fit(train_X_PMst,train_Y_PMst)
time_end = time.time()
print("time cost:%f s" % (time_end-time_start))

# 效果测试
y_pred = modelClass.model.predict(test_X_PMst)
print("Start 10-fold testing")
validation_r2 = cal_Rsqure(test_Y_PMst,y_pred)
print("R_squared: "+str(validation_r2))