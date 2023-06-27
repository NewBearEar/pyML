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

    #factory = ml_factory.ModelFactory('GTWGRNN')
    #modelClass = factory.getModelClassInstance()

    #gtwgrnnIns = modelClass.model
    #gtwgrnnIns.fit(trainset_fit[:,:-1],trainset_fit[:,-1])
    #res = gtwgrnnIns.predict(trainset_val[:,:-1],trainset_val[:,-1])

    #model = GWR(trainset_fit[:,1:3], trainset_fit[:,3:-1], trainset_fit[:,-1], 0.8, kernel='gaussian', fixed=True)



    u = np.array([(i - 1) % 12 for i in range(1, 1729)]).reshape(-1, 1)
    v = np.array([((i - 1) % 144) // 12 for i in range(1, 1729)]).reshape(-1, 1)
    t = np.array([(i - 1) // 144 for i in range(1, 1729)]).reshape(-1, 1)
    x1 = np.random.uniform(0, 1, (1728, 1))
    x2 = np.random.uniform(0, 1, (1728, 1))
    epsilon = np.random.randn(1728, 1)
    beta0 = 5
    beta1 = 3 + (u + v + t) / 6
    beta2 = 3 + ((36 - (6 - u) ** 2) * (36 - (6 - v) ** 2) * (36 - (6 - t) ** 2)) / 128
    y = beta0 + beta1 * x1 + beta2 * x2 + epsilon
    coords = np.hstack([u, v])
    X = np.hstack([x1, x2])
#    gtwr = GTWR(coords, t, X, y, 0.8, 1.9, kernel='gaussian', fixed=True).fit()

    #model = GTWR(trainset_fit[:, 1:3], trainset_fit[:, 0].reshape(-1, 1), trainset_fit[:, 3:-1], trainset_fit[:, -1],
    #             0.8, 0.01, kernel='gaussian', fixed=True)
#    model.fit()
    #res = model.predict(trainset_val[:, 1:3], trainset_val[:, 0], trainset_val[:, 3:-1])
    #return res

    fct = ml_factory.ModelFactory('GTWR')
    classIns = fct.getModelClassInstance()
    classIns.model.fit(trainset_fit[:, :-1], trainset_fit[:, -1])
    res = classIns.model.predict(trainset_val)
    return res
    """
    u = np.array([(i - 1) % 12 for i in range(1, 1729)]).reshape(-1, 1)
    v = np.array([((i - 1) % 144) // 12 for i in range(1, 1729)]).reshape(-1, 1)
    t = np.array([(i - 1) // 144 for i in range(1, 1729)]).reshape(-1, 1)
    x1 = np.random.uniform(0, 1, (1728, 1))
    x2 = np.random.uniform(0, 1, (1728, 1))
    epsilon = np.random.randn(1728, 1)
    beta0 = 5
    beta1 = 3 + (u + v + t) / 6
    beta2 = 3 + ((36 - (6 - u) ** 2) * (36 - (6 - v) ** 2) * (36 - (6 - t) ** 2)) / 128
    y = beta0 + beta1 * x1 + beta2 * x2 + epsilon
    coords = np.hstack([u, v])
    X = np.hstack([x1, x2])
    X2 = np.hstack([x1, x2])
    model = GWR(coords, X, y, 0.8, kernel='gaussian', fixed=True)
    res = model.predict(X2)
    print(1)
    return res
    """
print(testGWRFun())
