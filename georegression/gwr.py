from georegression.mgtwr.model import GWR
from typing import Union
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

class GWRAdaptor():
    def __init__(
            self,
            bandwidth: float = 0.8,
            kernel: str = 'gaussian',
            fixed: bool = True,
            constant: bool = True,
            thread: int = 1,
            convert: bool = False
    ):
        """
                Parameters
                ----------

                bandwidth     : scalar
                                bandwidth value consisting of either a distance or N
                                nearest neighbors; user specified or obtained using
                                sel

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

                thread        : int
                                The number of processes in parallel computation. If you have a large amount of data,
                                you can use it

                convert       : bool
                                Whether to convert latitude and longitude to plane coordinates.
        """
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.fixed = fixed
        self.constant = constant
        self.thread = thread
        self.convert = convert
        self.model = None

    def fit(self, X_train, y_train):
        """
                    Parameters
                    ----------
                    X_train : array-like (n_samples, n_features)
                    但必须保证前两个维度为 , 纬度 ，经度

                    y_train : array-like (n_samples,)
                    ------
                """
        self.coor = X_train[:, 0:2]
        self.X = X_train[:, 2:]
        self.y = y_train.reshape(-1, 1)
        self.model = GWR(self.coor, self.X, self.y,
                         bw=self.bandwidth, kernel=self.kernel, fixed=self.fixed, constant=self.constant,
                         thread=self.thread, convert=self.convert)
        #return self.model.fit()

    def predict(self, X_predict):
        res = self.model.predict(X_predict[:, 0:2], X_predict[:, 2:-1])
        return res
