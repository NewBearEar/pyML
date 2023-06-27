# -*- coding: utf-8 -*-
import numpy
import timeit

def read_global_data(dataPath):
    data_x = []
    data_y = []
    data_w = []
    fileIn = open(dataPath)
    for line in fileIn.readlines():
        if 'PM2.5' in line:
            continue  # 去掉表头
        lineArr = line.strip().split()
        if float(lineArr[4]) > -9999 and float(lineArr[5]) < 600:
            w = 1  # global model, w=1
            data_x.append(
                [float(lineArr[4]), float(lineArr[6]), float(lineArr[7]), float(lineArr[8]), float(lineArr[9]),
                 float(lineArr[10]), float(lineArr[11])])
            data_y.append(float(lineArr[5]))
            data_w.append(w)
    fileIn.close()
    return numpy.asarray(data_x), numpy.asarray(data_y), numpy.asarray(data_w)


def read_CV_data(dataPath):
    data_x = []
    data_y = []
    data_day = []
    data_lat = []
    data_lon = []
    fileIn = open(dataPath)
    for line in fileIn.readlines():
        if 'PM2.5' in line:
            continue
        lineArr = line.strip().split()
        lat = float(lineArr[2])
        lon = float(lineArr[3])
        time = lineArr[1]
        # DOY = int(time[5:8])
        # if float(lineArr[4]) > -9999 and (float(lineArr[5]) < 600) and (float(lineArr[18]) > -1):
        DOY = int(time[14:17])
        if float(lineArr[4]) > -9999 and (float(lineArr[5]) < 600):
            data_x.append(
                [float(lineArr[4]), float(lineArr[6]), float(lineArr[7]), float(lineArr[8]), float(lineArr[9]),
                 float(lineArr[10]), float(lineArr[11])])
            data_y.append(float(lineArr[5]))
            data_day.append(DOY)
            data_lat.append(lat)
            data_lon.append(lon)
    fileIn.close()
    return numpy.array(data_x), numpy.array(data_y), numpy.array(data_day), numpy.array(data_lat), numpy.array(data_lon)


def read_all_data(dataPath):
    data_x = []
    data_y = []
    data_day = []
    data_lat = []
    data_lon = []
    fileIn = open(dataPath)
    for line in fileIn.readlines():
        if 'PM2.5' in line:
            continue
        lineArr = line.strip().split()
        lat = float(lineArr[2])
        lon = float(lineArr[3])
        time = lineArr[1]
        DOY = int(time[14:17])
        data_x.append([float(lineArr[4]), float(lineArr[6]), float(lineArr[7]), float(lineArr[8]), float(lineArr[9]), float(lineArr[10]), float(lineArr[11])])
        data_y.append(float(lineArr[5]))
        data_day.append(DOY)
        data_lat.append(lat)
        data_lon.append(lon)
    fileIn.close()
    return numpy.array(data_x), numpy.array(data_y), numpy.array(data_day), numpy.array(data_lat), numpy.array(data_lon)


def calc_bandwidth_via_points(thistime, thislat, thislon, timeArr, latArr, lonArr, pointnum):
    disT = thistime - timeArr
    disS = calc_earthdis(thislat, thislon, latArr, lonArr)
    thisS = disS[disT == 0]
    sortS = numpy.sort(thisS)
    if len(thisS) > pointnum:
        fix_bandwidth = sortS[pointnum - 1]
    else:
        fix_bandwidth = numpy.max(sortS)
    return fix_bandwidth


def calc_wMatrix(thistime, thislat, thislon, timeArr, latArr, lonArr, bandwidth, lamda):
    disT = thistime - timeArr
    disS = calc_earthdis(thislat, thislon, latArr, lonArr)
    w = numpy.exp(-(disS * disS + lamda * (disT * disT)) / (bandwidth * bandwidth))
    w[disT < 0] = 0
    return w


def calc_earthdis(lat1, lon1, lat2, lon2):
    pos = numpy.abs(lat1 - lat2) + numpy.abs(lon1 - lon2) < 0.00001
    lat1 = float(lat1) / 180.0 * numpy.pi
    lon1 = float(lon1) / 180.0 * numpy.pi
    lat2 = lat2 / 180.0 * numpy.pi
    lon2 = lon2 / 180.0 * numpy.pi
    nEarthRadis = 6371.004
    angle = numpy.arccos(numpy.cos(lat1) * numpy.cos(lat2) * numpy.cos(lon1 - lon2) + numpy.sin(lat1) * numpy.sin(lat2))
    dis = nEarthRadis * angle
    dis[pos] = 0.0
    return dis


def calc_PMs(thistime, thislat, thislon, all_train_y, all_train_time, all_train_lat, all_train_lon, pointnum, dis):
    # calculating PMs for the validation sample
    disT = thistime - all_train_time
    thisPos = (disT == 0)
    this_train_y = all_train_y[thisPos]
    disS = calc_earthdis(thislat, thislon, all_train_lat[thisPos], all_train_lon[thisPos])
    disS = disS + 0.000001
    selected = (disS > dis)
    disS = disS[selected]
    if len(disS) < pointnum:
        pointnum = len(disS)
    selected_y = this_train_y[selected]
    sortS = numpy.sort(disS)
    poss = numpy.argsort(disS)
    mindis = sortS[0]
    val_PMs = numpy.sum(1 / (sortS[0: pointnum] * sortS[0: pointnum]) * selected_y[poss[0: pointnum]]) / numpy.sum(
        1 / (sortS[0: pointnum] * sortS[0: pointnum]))

    # calculating PMs for the fitting data set
    # train_PMs = []
    # for i in range(0, len(train_time)):
    #     disT2 = train_time[i] - all_train_time
    #     thisPos2 = (disT2 == 0)
    #     this_train_y2 = all_train_y[thisPos2]
    #     disS2 = calc_earthdis(train_lat[i], train_lon[i], all_train_lat[thisPos2], all_train_lon[thisPos2])
    #     selected2 = (disS2 >= mindis)
    #     disS2 = disS2[selected2]
    #     if len(disS2) < pointnum:
    #         pointnum = len(disS2)
    #     selected_y2 = this_train_y2[selected2]
    #     sortS2 = numpy.sort(disS2)
    #     poss2 = numpy.argsort(disS2)
    #     temp_PMs = numpy.sum(
    #         1 / (sortS2[0: pointnum] * sortS2[0: pointnum]) * selected_y2[poss2[0: pointnum]]) / numpy.sum(
    #         1 / (sortS2[0: pointnum] * sortS2[0: pointnum]))
    #     train_PMs.append(temp_PMs)

    return val_PMs, mindis


def calc_PMs_viaarray(thistime, thislat, thislon, train_time, train_lat, train_lon, all_train_y, all_train_time,
                      all_train_lat, all_train_lon, pointnum, dis):
    # calculating PMs for the validation sample
    disT = thistime - all_train_time
    thisPos = (disT == 0)
    this_train_y = all_train_y[thisPos]
    disS = calc_earthdis(thislat, thislon, all_train_lat[thisPos], all_train_lon[thisPos])
    selected = (disS >= dis)
    disS = disS[selected]
    if len(disS) < pointnum:
        pointnum = len(disS)
    selected_y = this_train_y[selected]
    sortS = numpy.sort(disS)
    poss = numpy.argsort(disS)
    mindis = sortS[0]
    val_PMs = numpy.sum(1 / (sortS[0: pointnum] * sortS[0: pointnum]) * selected_y[poss[0: pointnum]]) / numpy.sum(
        1 / (sortS[0: pointnum] * sortS[0: pointnum]))

    # calculating PMs for the fitting data set
    train_PMs = []
    num = len(train_time)
    train_time.shape = num, 1
    ZerosArr = numpy.zeros([1, len(all_train_time)])
    disTArr = train_time - all_train_time
    thisPosArr = disTArr == 0
    all_train_lat_Arr = all_train_lat - ZerosArr
    all_train_lon_Arr = all_train_lon - ZerosArr
    valid_all_train_lat_Arr = all_train_lat_Arr * thisPosArr
    valid_all_train_lon_Arr = all_train_lon_Arr * thisPosArr

    train_lat.shape = num, 1
    train_lat = train_lat - ZerosArr
    train_lon.shape = num, 1
    train_lon = train_lon - ZerosArr
    train_lat = train_lat / 180.0 * numpy.pi
    train_lon = train_lon / 180.0 * numpy.pi
    angle = numpy.arccos(numpy.cos(train_lat) * numpy.cos(valid_all_train_lat_Arr) * numpy.cos(
        train_lon - valid_all_train_lon_Arr) + numpy.sin(train_lat) * numpy.sin(valid_all_train_lat_Arr))
    nEarthRadis = 6371.004
    dis = nEarthRadis * angle

    for i in range(0, len(train_time)):
        disT2 = train_time[i] - all_train_time
        thisPos2 = (disT2 == 0)
        this_train_y2 = all_train_y[thisPos2]
        disS2 = calc_earthdis(train_lat[i], train_lon[i], all_train_lat[thisPos2], all_train_lon[thisPos2])
        selected2 = (disS2 >= mindis)
        disS2 = disS2[selected2]
        if len(disS2) < pointnum:
            pointnum = len(disS2)
        selected_y2 = this_train_y2[selected2]
        sortS2 = numpy.sort(disS2)
        poss2 = numpy.argsort(disS2)
        temp_PMs = numpy.sum(
            1 / (sortS2[0: pointnum] * sortS2[0: pointnum]) * selected_y2[poss2[0: pointnum]]) / numpy.sum(
            1 / (sortS2[0: pointnum] * sortS2[0: pointnum]))
        train_PMs.append(temp_PMs)

    return val_PMs, numpy.array(train_PMs)