import numpy as np
import math

class GTWGRNN():
    def __init__(self,bandWidth=4,bLambda=3,spread=0.1):
        self.bandWidth = bandWidth
        self.bLambda = bLambda * 10e4
        self.spread = spread
        self.X_train = None
        self.y_train = None

    def fit(self,X_train,y_train):
        """
            Parameters
            ----------
            X_train : array-like (n_samples, n_features)
            但必须保证前三个维度为 时间（day of year), 纬度 ，经度

            y_train : array-like (n_samples,)
            ------
        """
        self.X_train = X_train
        self.y_train = y_train
        self.timeLoc_train = X_train[:,0:3]
        self.xset_train = X_train[:,3:]

    def predict(self,X_predict,y_predict_real):
        """
            Parameters
            ----------
            X_predict : array-like (n_samples, n_features)
            待预测数据的自变量值
            但必须保证前三个维度为 时间（day of year), 纬度 ，经度

            y_real : array-like (n_samples,)
            待遇测数据的因变量 真实值
            ------
        """
        self.timeLoc_predict = X_predict[:,0:3]
        self.xset_predict = X_predict[:,3:]
        self.y_predict_real = y_predict_real
        y_predict_estimated = np.empty([y_predict_real.shape[0],1])

        for i in range(y_predict_real.shape[0]):

            fixed = self.__calcBandWidthByPoints(self.timeLoc_predict[i, 0], self.timeLoc_predict[i, 1], self.timeLoc_predict[0, 2],
                                                   self.timeLoc_train[:, 0], self.timeLoc_train[:, 1], self.timeLoc_train[:, 2], self.bandWidth)
            train_w = self.__calcWeightMatrix(self.timeLoc_predict[i, 0], self.timeLoc_predict[i, 1], self.timeLoc_predict[i, 2],
                                            self.timeLoc_train[:, 0], self.timeLoc_train[:, 1], self.timeLoc_train[:, 2], fixed, self.bLambda)
            pos = train_w > 0.000001
            if(pos.shape[0] < 2):
                y_predict_estimated[i] = -9999
                continue
            # setup bias and weights
            b = np.sqrt(-np.log(0.5))/self.spread
            i_p_w = self.xset_train[pos,:]
            p_ws_w = self.y_train[pos] * train_w[pos]
            p_s_w = train_w[pos]

            # predict
            disMatrix = np.sqrt(np.sum(np.power(i_p_w - self.xset_predict[i,:],2),axis=1))
            parttenLayer_input = disMatrix * b
            parttenLayer_output = np.exp(-(np.power(parttenLayer_input,2)))
            y = np.sum(parttenLayer_output * p_ws_w) / np.sum(parttenLayer_output * p_s_w)
            y_predict_estimated[i] = y
        return y_predict_estimated

    def __calcWeightMatrix(self,thisTime,thisLat,thisLon,time,lat,lon,bandWidth,bLambda):
        dis_T = thisTime - time
        dis_S = self.__calcEarthDis(thisLat,thisLon,lat,lon)
        w = np.exp(-(np.power(dis_S,2) + bLambda * np.power(dis_T,2)) / np.power(bandWidth,2) )
        w[dis_T<0] = 0
        return w

    def __calcBandWidthByPoints(self,thisTime,thisLat,thisLon,time,lat,lon,ptNum):
        dis_T = thisTime - time
        selected_pos = (np.abs(dis_T) < 0.000001)
        dis_S = self.__calcEarthDis(thisLat,thisLon,lat[selected_pos],lon[selected_pos])
        sorted_dis = np.sort(dis_S)
        if ptNum <= sorted_dis.shape[0]:
            bandWidth = sorted_dis[ptNum-1]
        else:
            bandWidth = np.max(sorted_dis)
        return bandWidth

    def __calcEarthDis(self,lat,lon,arrLat,arrLon):
        lat = lat / 180 * math.pi
        lon = lon / 180 * math.pi
        arrLat = arrLat / 180 * math.pi
        arrLon = arrLon / 180 * math.pi
        nEarthRadis = 6371.004
        angle = np.arccos(np.cos(lat) * np.cos(arrLat) * np.cos(lon-arrLon) + np.sin(lat) * np.sin(arrLat))
        return nEarthRadis * angle
