import ml_factory
from georegression.mgtwr.model import GWR,GTWR
import  pandas as pd
import  numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import numpy as np


def testGWRFun():
    data = pd.read_csv(r".\testData\grnn_test_data\data.csv",header=None)
    data = data.to_numpy()
    trainset_fit  = pd.read_csv(r".\testData\grnn_test_data\data_fit.csv",header=None).to_numpy()
    trainset_val = pd.read_csv(r".\testData\grnn_test_data\data_val.csv",header=None).to_numpy()
    scaler = MinMaxScaler()
    scaled_fit = scaler.fit_transform(np.array(trainset_fit[:,3:-1]))
    trainset_fit[:,3:-1] = scaled_fit
    scaled_val = scaler.fit_transform(np.array(trainset_val[:,3:-1]))
    trainset_val[:,3:-1] = scaled_val

    dropout = 0

    X = trainset_fit[:, 3:-2]
    y = trainset_fit[:, -2:]

    # 指定权重
    weight = [ 0.3 for i in range(y.shape[1])]

    fct = ml_factory.ModelFactory('MultiTaskDBN')
    classIns = fct.getModelClassInstance()
    classIns.model.fit(X, y, weight, dropout=dropout)
    X_predict = trainset_val[:,3:-2]
    res = classIns.model.predict(X_predict)
    return res

print(testGWRFun())
