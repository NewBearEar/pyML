# -*- coding: utf-8 -*-

from __future__ import print_function

from osgeo import gdal
import numpy
import theano
import theano.tensor as T
import cPickle
import os
import BasicFunc
import GTWNN


def files_in_folder(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.tif':
                L.append(os.path.join(root, file))
    return L


def read_singleband_tiff(inpath):
    ds = gdal.Open(inpath)
    col = ds.RasterXSize
    row = ds.RasterYSize
    geoTransform = ds.GetGeoTransform()
    proj = ds.GetProjection()
    data = numpy.zeros([row, col])
    dt = ds.GetRasterBand(1)
    data[:, :] = dt.ReadAsArray(0, 0, col, row)
    return data, geoTransform, proj


def write_singleband_tiff(outpath, data, geoTransform, proj):
    cols = data.shape[1]
    rows = data.shape[0]
    driver = gdal.GetDriverByName('Gtiff')
    outRaster = driver.Create(outpath, cols, rows, 1, gdal.GDT_Float64)
    outRaster.SetGeoTransform(geoTransform)
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(data)
    outRaster.SetProjection(proj)
    outRaster.FlushCache()


if __name__ == '__main__':
    train_dataset = r'E:\DataProducts\8_iGTWDNN\2015_revised_datarecords.txt'  # the data records path
    # all_train_x, all_train_y, all_train_time, all_train_lat, all_train_lon = BasicFunc.read_all_data(train_dataset)
    train_x, train_y, train_time, train_lat, train_lon = BasicFunc.read_CV_data(train_dataset)

    # network parameters
    n_hidden_size = [15]
    learning_rate = 0.01
    n_epochs = 120
    lamda = 80000.0
    bandwidth = 2
    FeatureNum = 7
    fnorm = open('models/dataNorm-global-15', 'r')
    s2 = fnorm.read()
    save_min_max_scaler = cPickle.loads(s2)
    fnet = open('models/net-global-15', 'rb')
    save_params = cPickle.load(fnet)

    # Step1:*************************************网络基本结构***********************************************************
    # 在NN构建花的时间最多，只构建一次，读取网络权值就行了，加快速度
    # allocate symbolic variables for the data
    index = T.lvector()
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.vector('y')  # the labels are presented as 1D vector of [int] labels
    geoW = T.vector('geoW')  #
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
        (param, param - learning_rate * gparam)
        for param, gparam in zip(regression.params, gparams)
    ]

    train_model = theano.function(
        inputs=[x, y, geoW],
        outputs=cost,
        updates=updates,
    )

    prediction_model = theano.function(
        inputs=[regression.input],
        outputs=regression.prediction
    )
    # *****************************************************************************************************************

    # step 2:prediction for every pixel in each image
    AOD_directory = r'E:\DataProducts\2_GeoiDBN_pm_china_GRL\GRL_revised\AODclip'
    RH_directory = r'E:\DataProducts\2_GeoiDBN_pm_china_GRL\GRL_revised\RH'
    WS_directory = r'E:\DataProducts\2_GeoiDBN_pm_china_GRL\GRL_revised\WS'
    TMP_directory = r'E:\DataProducts\2_GeoiDBN_pm_china_GRL\GRL_revised\TMP'
    PBL_directory = r'E:\DataProducts\2_GeoiDBN_pm_china_GRL\GRL_revised\PBL'
    PS_directory = r'E:\DataProducts\2_GeoiDBN_pm_china_GRL\GRL_revised\PS'
    NDVI_directory = r'E:\DataProducts\2_GeoiDBN_pm_china_GRL\GRL_revised\NDVI'
    OutputPM_directory = 'F:\PMretrieving'

    AOD_Files = files_in_folder(AOD_directory)
    RH_Files = files_in_folder(RH_directory)
    WS_Files = files_in_folder(WS_directory)
    TMP_Files = files_in_folder(TMP_directory)
    PBL_Files = files_in_folder(PBL_directory)
    PS_Files = files_in_folder(PS_directory)
    NDVI_Files = files_in_folder(NDVI_directory)

    AOD_FileCount = len(AOD_Files)

    for k in xrange(0, AOD_FileCount):
        AOD, aodGeoTransform, aodProj = read_singleband_tiff(AOD_Files[k])
        RH, rhGeoTransform, rhProj = read_singleband_tiff(RH_Files[k])
        WS, wsGeoTransform, wsProj = read_singleband_tiff(WS_Files[k])
        TMP, tmpGeoTransform, tmpProj = read_singleband_tiff(TMP_Files[k])
        PBL, pblGeoTransform, pblProj = read_singleband_tiff(PBL_Files[k])
        PS, psGeoTransform, psProj = read_singleband_tiff(PS_Files[k])
        NDVI, ndviGeoTransform, ndviProj = read_singleband_tiff(NDVI_Files[k])
        [m, n] = AOD.shape
        lat = numpy.zeros([m, n])
        lon = numpy.zeros([m, n])
        for mm in range(0, m):
            lat[mm, :] = aodGeoTransform[3] + 0.5 * aodGeoTransform[5] + mm * aodGeoTransform[5]  # lat = -0.1
        for nn in range(0, n):
            lon[:, nn] = aodGeoTransform[0] + 0.5 * aodGeoTransform[1] + nn * aodGeoTransform[1]

        nFileName = os.path.basename(AOD_Files[k])
        currentDOY = int(nFileName[14:17])

        lat_ = numpy.reshape(lat, (m * n, 1))
        lon_ = numpy.reshape(lon, (m * n, 1))
        AOD_ = numpy.reshape(AOD, (m * n, 1))
        RH_ = numpy.reshape(RH, (m * n, 1))
        WS_ = numpy.reshape(WS, (m * n, 1))
        TMP_ = numpy.reshape(TMP, (m * n, 1))
        PBL_ = numpy.reshape(PBL, (m * n, 1))
        PS_ = numpy.reshape(PS, (m * n, 1))
        NDVI_ = numpy.reshape(NDVI, (m * n, 1))
        pre_input = numpy.hstack((AOD_, RH_, WS_, TMP_, PBL_, PS_, NDVI_))

        valid_index = numpy.where(AOD_ > -9999.0)[0]
        valid_lat = lat_[valid_index]
        valid_lon = lon_[valid_index]
        valid_pre_input = pre_input[valid_index]
        pre_pm = []
        for kk in range(0, len(valid_lat)):
            # adaptive bandwidth
            # if abs(valid_lat[kk] - 39.15) > 0.001 or abs(valid_lon[kk] - 117.45) > 0.001:
            #     continue
            bd = BasicFunc.calc_bandwidth_via_points(currentDOY, valid_lat[kk], valid_lon[kk], train_time, train_lat,
                                                     train_lon, bandwidth)
            bd = bd if bd > 0 else 0.001
            # calculating the spatiotemporal weighting
            thisw = BasicFunc.calc_wMatrix(currentDOY, valid_lat[kk], valid_lon[kk], train_time, train_lat, train_lon,
                                           bd, lamda)
            included = thisw > 0.000001
            valid_train_w = thisw[included]
            valid_train_w = numpy.around(valid_train_w, decimals=6)  #地理权值小数点后6位的差异可带来反演值的改变
            valid_train_x = train_x[included]
            valid_train_y = train_y[included]

            this_train_x = valid_train_x
            this_pre_x = valid_pre_input[kk]
            this_pre_x.shape = 1, len(this_pre_x)

            print(k + 1, '/', AOD_FileCount, ':total=', len(valid_lat), 'pixel=', kk + 1, 'samples=',
                  len(valid_train_w), 'bandwidth=', bd)

            # ********************************微调网络**************************************************
            # load the global NN parameters for fine-tuning
            for j in range(len(n_hidden_size)):
                regression.hidden_layers[j].W.set_value(save_params[j * 2])
                regression.hidden_layers[j].b.set_value(save_params[j * 2 + 1])
            regression.LinearOutputLayer.W.set_value(save_params[-2])
            regression.LinearOutputLayer.b.set_value(save_params[-1])

            if this_train_x.shape[0] < 1:  # <6 samples are collected
                text_x = save_min_max_scaler.transform(this_pre_x)
                val_y_pred = prediction_model(text_x)
                print('sample<6, non-fine-tuning')
                continue

            # 归一化处理
            train_x_norm = numpy.asarray(save_min_max_scaler.transform(this_train_x))
            n_sample_num = train_x_norm.shape[0]
            batch_size = len(valid_train_y)
            n_train_batches = n_sample_num // batch_size
            FeatureNum = train_x_norm.shape[1]
            epoch = 1
            while epoch <= n_epochs:
                traincost = 0
                rr = numpy.arange(train_x_norm.shape[0])
                # numpy.random.shuffle(rr)
                for minibatch_index in xrange(n_train_batches):
                    train_minibatch_index = rr[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
                    traincost += train_model(train_x_norm[train_minibatch_index], valid_train_y[train_minibatch_index],
                                             valid_train_w[train_minibatch_index])
                # print('epoch=', epoch, 'cost=', traincost)
                epoch = epoch + 1
            test_x = save_min_max_scaler.transform(this_pre_x)
            val_y_pred = prediction_model(test_x)
            pre_pm.append(numpy.asscalar(val_y_pred))

        retrieved_pm = numpy.zeros([m, n]) - 9999.0
        retrieved_pm_ = numpy.reshape(retrieved_pm, (m * n, 1))
        pre_pm_toarray = numpy.array(pre_pm)
        pre_pm_toarray.shape = len(pre_pm), 1
        retrieved_pm_[valid_index] = numpy.array(pre_pm_toarray)
        retrieved_pm = numpy.reshape(retrieved_pm_, (m, n))

        write_singleband_tiff(OutputPM_directory + '\\' + os.path.basename(AOD_Files[k]), retrieved_pm, aodGeoTransform,
                              aodProj)
