from georegression.mgtwr.model import GTWR
from typing import Union
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
class GTWRAdaptor():
    def __init__(
            self,
            bandwidth: float = 0.8,
            tau: float = 0.01,
            kernel: str = 'gaussian',
            fixed: bool = True,
            constant: bool = True,
            thread: int = 1,
            convert: bool = False
    ):
        """
        Geographically and Temporally Weighted Regression

        Parameters
        ----------

        bw            : scalar
                        bandwidth value consisting of either a distance or N
                        nearest neighbors; user specified or obtained using
                        sel

        tau           : scalar
                        spatio-temporal scale

        kernel        : string
                        type of kernel function used to weight observations;
                        available options:
                        'gaussian'
                        'bisquare'
                        'exponential'

        fixed         : bool
                        True for distance based kernel function and  False for
                        adaptive (nearest neighbor) kernel function (default)

        constant      : bool
                        True to include intercept (default) in model and False to exclude
                        intercept.

        """

        self.bandwidth = bandwidth
        self.tau = tau
        self.kernel = kernel
        self.fixed = fixed
        self.constant = constant
        self.thread = thread
        self.convert = convert
        self.model = None

    def fit(self,X_train,y_train):
        """
                    Parameters
                    ----------
                    X_train : array-like (n_samples, n_features)
                    但必须保证前三个维度为 时间, 纬度 ，经度

                    y_train : array-like (n_samples,)
                    ------
                """
        self.t = X_train[:,0].reshape(-1,1)
        self.coor = X_train[:,1:3]
        self.X = X_train[:,3:]
        self.y = y_train.reshape(-1,1)
        self.model = GTWR(self.coor,self.t, self.X, self.y,
             bw = self.bandwidth,tau = self.tau,  kernel=self.kernel, fixed=self.fixed,constant=self.constant,thread=self.thread,convert=self.convert)
        #return self.model.fit()

    def predict(self,X_predict):

        res = self.model.predict(X_predict[:,1:3],X_predict[:,0].reshape(-1,1),X_predict[:,3:-1])
        return res

