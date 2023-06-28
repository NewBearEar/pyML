import ml_factory
from gc_gtwnn.gc_gtwnn_torch.GTWNN import GCGTWNN
import  pandas as pd
import  numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import numpy as np


def testGCGTWNNFun():
    data = pd.read_csv(r".\testData\grnn_test_data\data.csv",header=None)
    data = data.to_numpy()
    trainset_fit  = pd.read_csv(r".\testData\grnn_test_data\data_fit.csv",header=None).to_numpy()
    trainset_val = pd.read_csv(r".\testData\grnn_test_data\data_val.csv",header=None).to_numpy()
    scaler = MinMaxScaler()
    scaled_fit = scaler.fit_transform(np.array(trainset_fit[:,3:-1]))
    trainset_fit[:,3:-1] = scaled_fit
    scaled_val = scaler.fit_transform(np.array(trainset_val[:,3:-1]))
    trainset_val[:,3:-1] = scaled_val
    '''
    model = GCGTWNN(gc_epochs=10000,ft_epochs=1000)
    model.fit_gc(trainset_fit[:,1:3],trainset_fit[:,0],trainset_fit[:,3:-1],trainset_fit[:,-1])
    res = model.predict(trainset_val[:,1:3],trainset_val[:,0],trainset_val[:,3:-1],trainset_val[:,-1])
    '''

    factory = ml_factory.ModelFactory('GCGTWNN')
    modelClass = factory.getModelClassInstance()

    gcgtwnnIns = modelClass.model
    gcgtwnnIns.fit(trainset_fit[:, :-1], trainset_fit[:, -1])
    res = gcgtwnnIns.predict(trainset_val[:,:-1])

    return res

print(testGCGTWNNFun())