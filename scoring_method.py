import numpy as np

def cal_Rsqure(real_data,preout_data):
    # real_data:真实PM数据测试集
    # preout_data:预测PM数据
    # 计算R方
    real_data = real_data.flatten()
    preout_data = preout_data.flatten()
    fenzi = np.sum((real_data-preout_data)**2)
    fenmu = np.sum((real_data-np.mean(real_data))**2)
    Rsqure = 1-fenzi / fenmu
    return Rsqure
def cal_RMSE(real_data,preout_data):
    real_data = real_data.flatten()
    preout_data = preout_data.flatten()
    MSE = np.sum((real_data-preout_data)**2)/np.size(real_data,0)
    RMSE = MSE**0.5
    return RMSE

def cal_MAE(real_data,preout_data):
    # MAE 即 MPE
    real_data = real_data.flatten()
    preout_data = preout_data.flatten()
    MAE = np.sum(np.abs(real_data-preout_data))/np.size(real_data,0)
    return MAE

def cal_RPE(real_data,preout_data):
    real_data = real_data.flatten()
    preout_data = preout_data.flatten()
    aver = np.mean(real_data)
    RPE = cal_RMSE(real_data,preout_data)/aver
    return RPE