# -*- coding: utf-8 -*-

from __future__ import print_function

__docformat__ = 'restructedtext en'

import numpy
import timeit
import EvaluationIndex
import BasicFunc
import theano
import theano.tensor as T
import GTWNN
import pickle


def globalNN_Fitting(hidden_size, datasetpath):
    train_x, train_y, train_w = BasicFunc.read_global_data(datasetpath)
    temp_train_y, temp_train_y_pred, temp_val_y, temp_val_y_pred = GTWNN.train_nn(train_x, train_y, train_w,
                                                                                  train_x, train_y, hidden_size,
                                                                                  'global-15')
    train_Rsquare = EvaluationIndex.calc_Rsquare(temp_train_y, temp_train_y_pred)
    train_RMSE = EvaluationIndex.calc_RMSE(temp_train_y, temp_train_y_pred)
    train_MPE = EvaluationIndex.calc_MPE(temp_train_y, temp_train_y_pred)
    train_RPE = EvaluationIndex.calc_RPE(temp_train_y, temp_train_y_pred)
    print('trainR2=', train_Rsquare, ' trainRMSE=', train_RMSE, ' trainMPE=', train_MPE, ' train_RPE=', train_RPE)


def globalNN_CV(hidden_size, directory, modelname):
    val_allobserved = []
    val_allestimated = []
    for nRound in range(1, 11):
        train_dataset = directory + '\Fitting_' + str(nRound) + '.txt'
        val_dataset = directory + '\Val_' + str(nRound) + '.txt'
        train_x, train_y, train_time, train_lat, train_lon = BasicFunc.read_CV_data(train_dataset)
        val_x, val_y, val_time, val_lat, val_lon = BasicFunc.read_CV_data(val_dataset)

        train_w = numpy.zeros(len(train_y)) + 1
        temp_train_y, temp_train_y_pred, temp_val_y, temp_val_y_pred = GTWNN.train_nn(train_x, train_y, train_w, val_x,
                                                                                      val_y, hidden_size,
                                                                                      modelname + str(nRound))
        val_allobserved = numpy.hstack((val_allobserved, temp_val_y.transpose()))
        val_allestimated = numpy.hstack((val_allestimated, temp_val_y_pred.transpose()))

    val_Rsquare = EvaluationIndex.calc_Rsquare(val_allobserved, val_allestimated)
    val_RMSE = EvaluationIndex.calc_RMSE(val_allobserved, val_allestimated)
    val_MPE = EvaluationIndex.calc_MPE(val_allobserved, val_allestimated)
    val_RPE = EvaluationIndex.calc_RPE(val_allobserved, val_allestimated)
    print('num=', len(val_allobserved), ' valR2=', val_Rsquare, ' valRMSE=', val_RMSE, ' valMPE=', val_MPE, ' valRPE=',
          val_RPE)
    strResults = 'num=' + str(len(val_allobserved)) + ' valR2=' + str(val_Rsquare) + ' valRMSE=' + str(
        val_RMSE) + ' valMPE=' + str(val_MPE) + ' valRPE=' + str(val_RPE)
    return strResults

def iGTWNN_CV(hidden_size, lamda, bandwidth, n_epochs, directory, modelname):
    # Step1:*************************************网络基本结构***********************************************************
    # 在NN构建花的时间最多，只构建一次，读取网络权值就行了，加快速度
    n_hidden_size = hidden_size
    FeatureNum = 7
    # allocate symbolic variables for the data
    index = T.lvector()
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.vector('y')  # the labels are presented as 1D vector of [int] labels
    geoW = T.vector('geoW')  #
    learningrate = T.scalar('lr')
    rng = numpy.random.RandomState(1234)
    # construct the MLP class
    regression = GTWNN.NN(
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
        inputs=[x, y, geoW, learningrate],
        outputs=cost,
        updates=updates,
    )

    prediction_model = theano.function(
        inputs=[regression.input],
        outputs=regression.prediction
    )
    # *****************************************************************************************************************

    # 开始验证
    point_num = 10
    val_allobserved = []
    val_allestimated = []
    val_alltime = []
    val_alllat = []
    val_alllon = []
    val_allsamplenum = []
    val_interpolation = []
    temptest = []
    for nRound in range(1, 2):
        # load saved parameters
        fnorm = open('models/dataNorm-' + modelname + str(nRound), 'r')
        s2 = fnorm.read()
        save_min_max_scaler = pickle.loads(s2)
        fnet = open('models/net-' + modelname + str(nRound), 'rb')
        save_params = pickle.load(fnet)

        train_dataset = directory + '\Fitting_' + str(nRound) + '.txt'
        val_dataset = directory + '\Val_' + str(nRound) + '.txt'
        train_x, train_y, train_time, train_lat, train_lon = BasicFunc.read_CV_data(train_dataset)
        val_x, val_y, val_time, val_lat, val_lon = BasicFunc.read_CV_data(val_dataset)

        # 对于每一个验证点来说
        for i in range(0, len(val_y)):  # this: val_time[i], val_lat[i], val_lon[i]
            # adaptive bandwidth
            bd = BasicFunc.calc_bandwidth_via_points(val_time[i], val_lat[i], val_lon[i], train_time, train_lat,
                                                     train_lon, bandwidth)
            # calculating the spatiotemporal weighting
            thisw = BasicFunc.calc_wMatrix(val_time[i], val_lat[i], val_lon[i], train_time, train_lat, train_lon,
                                           bd, lamda)
            included = thisw > 0.000001
            valid_train_w = thisw[included]
            valid_train_w = numpy.around(valid_train_w, decimals=6)
            valid_train_x = train_x[included]
            valid_train_y = train_y[included]

            this_train_x = valid_train_x
            this_val_x = val_x[i, :]
            this_val_x.shape = 1, len(this_val_x)

            #  微调local NN
            start_time2 = timeit.default_timer()
            # ********************************微调网络**************************************************
            # load the global NN parameters for fine-tuning
            for l in range(len(n_hidden_size)):
                regression.hidden_layers[l].W.set_value(save_params[l * 2])
                regression.hidden_layers[l].b.set_value(save_params[l * 2 + 1])
            regression.LinearOutputLayer.W.set_value(save_params[-2])
            regression.LinearOutputLayer.b.set_value(save_params[-1])

            # the weights are set as 0
            # for jj in range(len(n_hidden_size)):
            #     temp = save_params[jj * 2]
            #     temp[temp < 1000000] = 0
            #     regression.hidden_layers[jj].W.set_value(temp)
            #     temp = save_params[jj * 2 + 1]
            #     temp[temp < 1000000] = 0
            #     regression.hidden_layers[jj].b.set_value(temp)
            # temp = save_params[-2]
            # temp[temp < 1000000] = 0
            # regression.LinearOutputLayer.W.set_value(temp)
            # temp = save_params[-1]
            # temp[temp < 1000000] = 0
            # regression.LinearOutputLayer.b.set_value(temp)

            if this_train_x.shape[0] < 1:  # <6 samples are collected
                text_x = save_min_max_scaler.transform(this_val_x)
                val_y_pred = prediction_model(text_x)
                val_allestimated.append(val_y_pred.item())
                val_allobserved.append(val_y[i])
                val_alltime.append(val_time[i])
                val_alllat.append(val_lat[i])
                val_alllon.append(val_lon[i])
                print('sample<2, non-fine-tuning')
                continue

            # 归一化处理
            train_x_norm = numpy.asarray(save_min_max_scaler.transform(this_train_x))
            n_sample_num = train_x_norm.shape[0]
            # batch_size = 1
            batch_size = len(valid_train_y)
            n_train_batches = n_sample_num // batch_size
            FeatureNum = train_x_norm.shape[1]

            # print('... training')
            InitialLearningRate = 0.01
            lr = InitialLearningRate
            epoch = 1
            while epoch <= n_epochs:
                traincost = 0
                kk = numpy.arange(train_x_norm.shape[0])
                numpy.random.shuffle(kk)
                for minibatch_index in range(n_train_batches):
                    # The learning rate can be delayed according to 'poly'
                    # lr = InitialLearningRate * (1 - float(n_train_batches * (epoch - 1) + minibatch_index + 1) / float(
                    #     n_epochs * n_train_batches)) ** 0.9
                    train_minibatch_index = kk[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
                    traincost += train_model(train_x_norm[train_minibatch_index], valid_train_y[train_minibatch_index],
                                             valid_train_w[train_minibatch_index], lr)
                    # text_x1 = save_min_max_scaler.transform(this_val_x)
                    # val_y_pred1 = numpy.asscalar(prediction_model(text_x1))
                # print('epoch = ', epoch, ' train_cost = ', traincost)
                epoch = epoch + 1
            text_x = save_min_max_scaler.transform(this_val_x)
            val_y_pred = prediction_model(text_x)
            val_allestimated.append(val_y_pred.item())
            # *********************************************************************************************************
            end_time2 = timeit.default_timer()
            val_allobserved.append(val_y[i])
            val_alltime.append(val_time[i])
            val_alllat.append(val_lat[i])
            val_alllon.append(val_lon[i])
            val_allsamplenum.append(len(valid_train_w))
            print(nRound, ' round ', i, ' sample: samples=', n_sample_num, ' NN_time=', (end_time2 - start_time2), ' bandwidth=', bd)

    val_allobserved = numpy.array(val_allobserved)
    val_allestimated = numpy.array(val_allestimated)
    val_Rsquare = EvaluationIndex.calc_Rsquare(val_allobserved, val_allestimated)
    val_RMSE = EvaluationIndex.calc_RMSE(val_allobserved, val_allestimated)
    val_MPE = EvaluationIndex.calc_MPE(val_allobserved, val_allestimated)
    val_RPE = EvaluationIndex.calc_RPE(val_allobserved, val_allestimated)
    slope, intercept = EvaluationIndex.calc_slope(val_allobserved, val_allestimated)
    strResults = 'epoch=' + str(n_epochs) + ' num=' + str(len(
        val_allobserved)) + ': valR2=' + '%.4f' % val_Rsquare + ' valRMSE=' + '%.4f' % val_RMSE + ' valMPE=' + '%.4f' % val_MPE + ' val_RPE=' + '%.4f' % val_RPE + ' slope=' + str(
        slope)
    print(strResults)
    return strResults


if __name__ == '__main__':
    # establishing global NN
    hidden_size = [15]
    # dataPath = r'E:\DataProducts\8_iGTWDNN\2015_revised_datarecords.txt'
    # globalNN_Fitting(hidden_size, dataPath)

    name = 'siteCV'
    modelname = name + '-15-round'

    # model CV process
    # # step 1: establishing global NN
    # directory = 'E:\\DataProducts\\8_STNNF\\' + name
    # strGlobalResults = globalNN_CV(hidden_size, directory, modelname)
    # fsave = open('results_save_in_txt.txt', 'a+')
    # fsave.write(name + '\n' + 'global:' + strGlobalResults + '\n')
    # fsave.close()
    
    # # step 2: fine-tuning the local NN based on global NN
    directory = 'E:\\DataProducts\\8_STNNF\\' + name   #修改为实际数据所存放的路径
    Finetuning_epochs = [120]
    lamda = [80000.0]
    bandwidth = [2]
    for k in range(0, len(Finetuning_epochs)):
        for i in range(0, len(lamda)):
            for j in range(0, len(bandwidth)):
                start_time = timeit.default_timer()
                strResults = iGTWNN_CV(hidden_size, lamda[i], bandwidth[j], Finetuning_epochs[k], directory, modelname)
                end_time = timeit.default_timer()
                fsave = open('results_save_in_txt.txt', 'a+')
                fsave.write('hidden_size=' + str(hidden_size) + ' lamda=' + str(lamda[i]) + ' bandwidth=' + str(
                    bandwidth[j]) + ' ' + strResults + ' time=' + str(end_time - start_time) + '\n')
                fsave.close()

