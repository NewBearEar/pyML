import numpy as np
from sklearn import linear_model


def calc_Rsquare(data1, data2):
    R = np.corrcoef(data1, data2)
    return R[0, 1] * R[0, 1]


def calc_RMSE(data1, data2):
    aver = np.mean(np.power(data1 - data2, 2))
    return np.sqrt(aver)


def calc_MPE(data1, data2):
    return np.mean(np.abs(data2 - data1))


def calc_RPE(data1, data2):
    return calc_RMSE(data1, data2) / np.mean(data1)


def calc_slope(data1, data2):
    data1.shape = len(data1), 1
    data2.shape = len(data2), 1
    regr = linear_model.LinearRegression()
    regr.fit(data1, data2)
    a, b = regr.coef_, regr.intercept_
    return a, b
