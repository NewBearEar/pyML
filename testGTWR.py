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

    fct = ml_factory.ModelFactory('GTWR')
    classIns = fct.getModelClassInstance()
    classIns.model.fit(trainset_fit[:, :-1], trainset_fit[:, -1])
    res = classIns.model.predict(trainset_val)
    return res

print(testGWRFun())
