import  gtwgrnn.gtwgrnn
import  pandas as pd
import  numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler

def testFun():
    data = pd.read_csv(r"E:\PyProjects\pyMLMethods_Test_v2.0\testData\grnn_test_data\data.csv",header=None)
    data = data.to_numpy()
    trainset_fit  = pd.read_csv(r"E:\PyProjects\pyMLMethods_Test_v2.0\testData\grnn_test_data\data_fit.csv",header=None).to_numpy()
    trainset_val = pd.read_csv(r"E:\PyProjects\pyMLMethods_Test_v2.0\testData\grnn_test_data\data_val.csv",header=None).to_numpy()
    scaler = MinMaxScaler()
    scaled_fit = scaler.fit_transform(np.array(trainset_fit[:,3:-1]))
    trainset_fit[:,3:-1] = scaled_fit
    scaled_val = scaler.fit_transform(np.array(trainset_val[:,3:-1]))
    trainset_val[:,3:-1] = scaled_val


    gtwgrnnIns = gtwgrnn.gtwgrnn.GTWGRNN()
    gtwgrnnIns.fit(trainset_fit[:,:-1],trainset_fit[:,-1])
    res = gtwgrnnIns.predict(trainset_val[:,:-1],trainset_val[:,-1])


    return res

print(testFun())