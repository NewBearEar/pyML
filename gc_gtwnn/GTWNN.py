# -*- coding: utf-8 -*-

from __future__ import print_function

__docformat__ = 'restructedtext en'

import numpy
import theano
import timeit
import theano.tensor as T
from sklearn import preprocessing
import EvaluationIndex
import pickle
import BasicFunc

class LinearOutputLayer(object):
    def __init__(self, input, n_in, n_out):
        if n_out == 1:
            n_inValue = numpy.zeros(n_in, dtype=theano.config.floatX)
            n_outValue = 0.
        else:
            n_inValue = numpy.zeros((n_in, n_out), dtype=theano.config.floatX)
            n_outValue = numpy.zeros(n_out, dtype=theano.config.floatX)

        self.W = theano.shared(
                value=n_inValue,
                name='W',
                borrow=True
        )

        self.b = theano.shared(
                value=n_outValue,
                name='b',
                borrow=True
        )
        self.p_y_given_x = T.dot(input, self.W) + self.b
        self.y_pred = self.p_y_given_x
        self.params = [self.W, self.b]
        self.input = input

    def Loss(self, y, geoW):
        # return T.mean(((y- self.p_y_given_x)**2)[T.arange(y.shape[0])])
        return 0.5 * T.mean(geoW * T.pow((y - self.p_y_given_x), 2))


# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):

        self.input = input
        if W is None:
            W_values = numpy.asarray(
                    rng.uniform(
                            low=-numpy.sqrt(6. / (n_in + n_out)),
                            high=numpy.sqrt(6. / (n_in + n_out)),
                            size=(n_in, n_out)
                    ),
                    dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the models
        self.params = [self.W, self.b]


# start-snippet-2
class NN(object):
    def __init__(self, rng, input, n_in, n_hiddenlayers_size, n_out):

        self.hidden_layers = []
        self.params = []
        self.n_hiddenlayers = len(n_hiddenlayers_size)
        L1_w_sum = 0
        L2_w_sum = 0
        #assert self.hidden_layers > 0

        for i in range(self.n_hiddenlayers):
            if i == 0:
                input_size = n_in
            else:
                input_size = n_hiddenlayers_size[i - 1]

            if i == 0:
                layer_input = input
            else:
                layer_input = self.hidden_layers[-1].output

            hiddenLayer = HiddenLayer(
                    rng=rng,
                    input=layer_input,
                    n_in=input_size,
                    n_out=n_hiddenlayers_size[i],
                    activation=T.nnet.sigmoid
            )
            self.hidden_layers.append(hiddenLayer)
            self.params.extend(hiddenLayer.params)
            L1_w_sum += abs(hiddenLayer.W).sum()
            L2_w_sum += abs(hiddenLayer.W ** 2).sum()

        self.LinearOutputLayer = LinearOutputLayer(
                input=self.hidden_layers[-1].output,
                n_in=n_hiddenlayers_size[-1],
                n_out=n_out
        )

        self.params.extend(self.LinearOutputLayer.params)

        self.L1 = (
            L1_w_sum + abs(self.LinearOutputLayer.W).sum()
        )
        self.L2_sqr = (
            L2_w_sum + (self.LinearOutputLayer.W ** 2).sum()
        )
        self.MLPLoss = (
            self.LinearOutputLayer.Loss
        )
        self.input = input
        self.prediction = (
            self.LinearOutputLayer.y_pred
        )


def train_nn(train_x, train_y, train_w, val_x, val_y, hidden_size, save_name):
    InitialLearningRate = 0.01
    L1_reg = 0.001
    L2_reg = 0.0001
    n_epochs = 800
    batch_size = 100
    n_hidden_size = hidden_size

    # 归一化处理
    min_max_scaler = preprocessing.MinMaxScaler()
    train_x_norm = numpy.asarray(min_max_scaler.fit_transform(train_x))

    train_set_x = theano.shared(
            numpy.asarray(train_x_norm,
                          dtype=theano.config.floatX),
            borrow=True)
    train_set_y = theano.shared(
            numpy.asarray(train_y,
                          dtype=theano.config.floatX),
            borrow=True)

    n_sample_num = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches = n_sample_num // batch_size
    FeatureNum = train_set_x.get_value(borrow=True).shape[1]

    print('   样本数:', n_sample_num, '   变量数:', FeatureNum)

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    # print('... building the models')

    # allocate symbolic variables for the data
    index = T.lvector()
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.vector('y')  # the labels are presented as 1D vector of [int] labels
    geoW = T.vector('geoW')  #
    learningrate = T.scalar('lr')

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    regression = NN(
            rng=rng,
            input=x,
            n_in=FeatureNum,
            n_hiddenlayers_size=n_hidden_size,
            n_out=1
    )

    cost = (
        regression.MLPLoss(y, geoW)
                # + L1_reg * regression.L1
                # + L2_reg * regression.L2_sqr
    )

    gparams = [T.grad(cost, param) for param in regression.params]
    updates = [
        (param, param - learningrate * gparam)
        for param, gparam in zip(regression.params, gparams)
        ]

    train_model = theano.function(
            inputs=[index, geoW, learningrate],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index],
                y: train_set_y[index]
            }
    )

    prediction_model = theano.function(
            inputs=[regression.input],
            outputs=regression.prediction
    )

    # print('... training')
    epoch = 1
    lr = InitialLearningRate
    while (epoch <= n_epochs):
        cost = 0
        kk = numpy.arange(train_set_x.get_value(borrow=True).shape[0])
        numpy.random.shuffle(kk)
        for minibatch_index in xrange(n_train_batches):
            # lr = InitialLearningRate * (1 - float(n_train_batches * (epoch - 1) + minibatch_index + 1) / float(
            #     n_epochs * n_train_batches)) ** 0.9
            train_minibatch_index = kk[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            cost += train_model(train_minibatch_index, train_w[train_minibatch_index], lr)
        # train_y_pred = prediction_model(X_train_minmax)
        # train_Rsquare = EvaluationIndex.calc_Rsquare(numpy.array(train_y).transpose(), numpy.array(train_y_pred).transpose())
        # train_RMSE = EvaluationIndex.calc_RMSE(numpy.array(train_y).transpose(), numpy.array(train_y_pred).transpose())
        # text_X = min_max_scaler.transform(val_x)
        # val_y_pred = prediction_model(text_X)
        # val_Rsquare = EvaluationIndex.calc_Rsquare(numpy.array(val_y).transpose(), numpy.array(val_y_pred).transpose())
        # val_RMSE = EvaluationIndex.calc_RMSE(numpy.array(val_y).transpose(), numpy.array(val_y_pred).transpose())
        # print('epoch:', epoch, '  error:', cost / n_train_batches, '  train_R2=', train_Rsquare, '  train_RMSE=', train_RMSE, '  test_R2=', val_Rsquare, '  test_RMSE=', val_RMSE)
        epoch = epoch + 1
    # this_train_x = X_train_minmax[train_w == 1] #当前块的拟合结果
    pre_y = prediction_model(train_x_norm)
    # this_train_y = train_y[train_w == 1]
    train_Rsquare = EvaluationIndex.calc_Rsquare(numpy.array(train_y).transpose(), numpy.array(pre_y).transpose())
    train_RMSE = EvaluationIndex.calc_RMSE(numpy.array(train_y).transpose(), numpy.array(pre_y).transpose())

    text_X = min_max_scaler.transform(val_x)
    val_y_pred = prediction_model(text_X)
    val_Rsquare = EvaluationIndex.calc_Rsquare(numpy.array(val_y).transpose(), numpy.array(val_y_pred).transpose())
    val_RMSE = EvaluationIndex.calc_RMSE(numpy.array(val_y).transpose(), numpy.array(val_y_pred).transpose())
    # print(save_name, '     local validation: trainR2=',train_Rsquare, ' trainRMSE=',train_RMSE, 'valR2=',val_Rsquare, ' valRMSE=',val_RMSE)

    # save the parameters for neural network
    f1 = file('models/net-' + save_name, 'wb')
    pickle.dump([param1.get_value() for param1 in regression.params], f1, protocol=pickle.HIGHEST_PROTOCOL)
    f1.close()

    # save the parameters for data normalization
    s = pickle.dumps(min_max_scaler)
    f2 = open('models/dataNorm-' + save_name, 'w')
    f2.write(s)
    f2.close()

    return numpy.array(train_y), numpy.array(pre_y), numpy.array(val_y), numpy.array(val_y_pred)


def finetune_nn(train_x, train_y, train_w, val_x, hidden_size):
    learning_rate = 0.01
    # L1_reg = 0.001
    # L2_reg = 0.0001
    n_epochs = 80
    batch_size = len(train_y)
    n_hidden_size = hidden_size

    # 归一化处理
    # load Xsetting
    f2 = open('models/dataNorm-dis50-round1', 'r')
    s2 = f2.read()
    min_max_scaler = pickle.loads(s2)
    train_x_norm = numpy.asarray(min_max_scaler.transform(train_x))

    train_set_x = theano.shared(
            numpy.asarray(train_x_norm,
                          dtype=theano.config.floatX),
            borrow=True)
    train_set_y = theano.shared(
            numpy.asarray(train_y,
                          dtype=theano.config.floatX),
            borrow=True)

    n_sample_num = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches = n_sample_num // batch_size
    FeatureNum = train_set_x.get_value(borrow=True).shape[1]

    print('   样本数:', n_sample_num, '   变量数:', FeatureNum)

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    # print('... building the models')

    # allocate symbolic variables for the data
    index = T.lvector()
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.vector('y')  # the labels are presented as 1D vector of [int] labels
    geoW = T.vector('geoW')  #

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    regression = NN(
            rng=rng,
            input=x,
            n_in=FeatureNum,
            n_hiddenlayers_size=n_hidden_size,
            n_out=1
    )

    cost = (
        regression.MLPLoss(y, geoW)
                # + L1_reg * regression.L1
                # + L2_reg * regression.L2_sqr
    )

    gparams = [T.grad(cost, param) for param in regression.params]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(regression.params, gparams)
        ]

    train_model = theano.function(
            inputs=[index, geoW],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index],
                y: train_set_y[index]
            }
    )

    prediction_model = theano.function(
            inputs=[regression.input],
            outputs=regression.prediction
    )

    # load the global NN parameters for fine-tuning
    f = open('models/net-dis50-round1', 'rb')
    params = pickle.load(f)
    for i in range(len(n_hidden_size)):
        regression.hidden_layers[i].W.set_value(params[i * 2])
        regression.hidden_layers[i].b.set_value(params[i * 2 + 1])
    regression.LinearOutputLayer.W.set_value(params[-2])
    regression.LinearOutputLayer.b.set_value(params[-1])

    # print('... training')
    epoch = 1
    while (epoch <= n_epochs):
        cost = 0
        kk = numpy.arange(train_set_x.get_value(borrow=True).shape[0])
        numpy.random.shuffle(kk)
        for minibatch_index in xrange(n_train_batches):
            train_minibatch_index = kk[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            cost += train_model(train_minibatch_index, train_w[train_minibatch_index])
        # print('cost = ', cost)
        epoch = epoch + 1

    text_x = min_max_scaler.transform(val_x)
    val_y_pred = prediction_model(text_x)
    return numpy.asscalar(val_y_pred)


def sim_nn(input_x, hidden_size):
    n_hidden_size = hidden_size
    FeatureNum = input_x.shape[1]
    x = T.matrix('x')
    rng = numpy.random.RandomState(1234)
    # construct the MLP class
    regression = NN(
            rng=rng,
            input=x,
            n_in=FeatureNum,
            n_hiddenlayers_size=n_hidden_size,
            n_out=1
    )
    prediction_model = theano.function(
            inputs=[regression.input],
            outputs=regression.prediction
    )

    # load Xsetting
    f2 = open('models/dataNorm-global', 'r')
    s2 = f2.read()
    min_max_scaler = pickle.loads(s2)
    input_x_norm = numpy.asarray(min_max_scaler.transform(input_x))

    # load the global NN parameters for fine-tuning
    f = open('models/net-global', 'rb')
    params = pickle.load(f)
    for i in range(len(n_hidden_size)):
        regression.hidden_layers[i].W.set_value(params[i * 2])
        regression.hidden_layers[i].b.set_value(params[i * 2 + 1])
    regression.LinearOutputLayer.W.set_value(params[-2])
    regression.LinearOutputLayer.b.set_value(params[-1])

    y_pred = prediction_model(input_x_norm)
    return numpy.array(y_pred)


def predict_nn( n_hidden_size):
    dataset=r'E:\DataProducts\4_realtime_pm_whc\AODPM\0records\2016_whc_RT_records.txt'
    val_x, val_y = BasicFunc.read_val_data(dataset, 32.3, 112.0, 28.4, 116.7)

    FeatureNum = val_x.shape[1]
    x = T.matrix('x')
    rng = numpy.random.RandomState(1234)
    regression = NN(
            rng=rng,
            input=x,
            n_in=FeatureNum,
            n_hiddenlayers_size=n_hidden_size,
            n_out=1
    )
    prediction_model = theano.function(
            inputs=[regression.input],
            outputs=regression.prediction
    )
    # load Xsetting
    f2=open('models/trained_Xsetting','r')
    s2=f2.read()
    min_max_scaler=pickle.loads(s2)

    # load the parameters of NN
    f = open('models/trained_BPNN', 'rb')
    params = pickle.load(f)
    for i in range(len(n_hidden_size)):
        regression.hidden_layers[i].W.set_value(params[i * 2])
        regression.hidden_layers[i].b.set_value(params[i * 2 + 1])
    regression.LinearOutputLayer.W.set_value(params[-2])
    regression.LinearOutputLayer.b.set_value(params[-1])

    # prediction on the test dataset
    text_X = min_max_scaler.transform(val_x)
    y_pred = prediction_model(text_X)
    Rsquare = EvaluationIndex.calc_Rsquare(numpy.array(val_y).transpose(), numpy.array(y_pred).transpose())
    RMSE = EvaluationIndex.calc_RMSE(numpy.array(val_y).transpose(), numpy.array(y_pred).transpose())
    print(' R2=',Rsquare,' RMSE=',RMSE)



