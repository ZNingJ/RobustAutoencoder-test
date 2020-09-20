import datetime

import PIL.Image as Image
from data import ImShow as I
import numpy as np
import tensorflow as tf
from model import l21RobustDeepAutoencoderOnST as l21RDA
import os
from collections import Counter
from sklearn.metrics import precision_score as precision
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score as recall
from sklearn.metrics import confusion_matrix as CM
import matplotlib.pyplot as plt
import pandas as pd


def l21RDAE(X, layers, lamda, folder, learning_rate=0.15, inner=100, outer=10, batch_size=133, inputsize=(28, 28)):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    os.chdir(folder)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            rael21 = l21RDA.RobustL21Autoencoder(sess=sess, lambda_=lamda * X.shape[0], layers_sizes=layers)
            l21L, l21S = rael21.fit(X=X, sess=sess, inner_iteration=inner, iteration=outer, batch_size=batch_size,
                                    learning_rate=learning_rate, verbose=True)
            l21R = rael21.getRecon(X=X, sess=sess)
            l21H = rael21.transform(X, sess)
            l21S.dump("l21S.npk")
    os.chdir("../")


def experiment_frame(X,elem_num,lamda_list):
    inner = 100
    outer = 8

    layers = [elem_num, 400, 200]  ## S trans
    folder = r"OutlierDetectionResult"
    if not os.path.isdir(folder):
        os.makedirs(folder)
    os.chdir(folder)

    for lam in lamda_list:
        folder = "lam" + str(lam)
        l21RDAE(X=X, layers=layers, lamda=lam, folder=folder, learning_rate=0.005,
                inner=inner, outer=outer, batch_size=133, inputsize=(476, 166))
    os.chdir("../")


def binary_y(value):
    if value == 23:
        return "o"
    else:
        return "m"

def binary_y2(value):
    if value == 3:
        return "o"
    else:
        return "m"

def binary_y3(value):
    if value == 0:
        return "m"
    else:
        return "o"

def binary_y4(value):
    if value == 5:
        return "o"
    else:
        return "m"

def binary_y5(value):
    if value == 7:
        return "o"
    else:
        return "m"

def binary_y6(value):
    if value == 1:
        return "m"
    else:
        return "o"

if __name__ == "__main__":

    folder = 'OutlierDetectionResult'
    for n in range(1,8):
        dataset = n
        # 1-ISOLET,2-MF-3,3-Arrhythmia,4-MF-5,5-MF-7,6-ionosphere,7-Musk2
        if dataset==1:
            elem_num=617
            filename=r"data/ISOLET-23/data_23.dat"
            print('当前数据集是：{0}'.format(filename))
            t1 = datetime.datetime.now()
            print('从当前时间开始:{0}'.format(t1))
            X = pd.read_csv(filename, header=None, index_col=None, skiprows=0, sep=',')
            X = X.iloc[:,:617].values

            lambda_list = [0.001]   #0.0017,0.0018  ,0.0015
            experiment_frame(X,elem_num,lambda_list)

            lam_list = list(map(str, lambda_list))
            y_loc = r"data/ISOLET-23/classid_23.dat"
            y = pd.read_csv(y_loc, header=None, index_col=None, skiprows=0, sep=',')
            y = y.iloc[:,0].values
            print(Counter(y))
            bi_y = list(map(binary_y, y))
            print(Counter(bi_y))

            for i, lam in enumerate(lam_list):
                print("lambda:", lam)
                print('bi_y:{0}'.format(bi_y))
                print('bi_y:{0}'.format(Counter(bi_y)))
                S = np.load(folder + "\\" + "lam" + lam + "\\" + r"l21S.npk", allow_pickle=True)
                zscore = np.linalg.norm(S, axis=1)
                print('score:{0}'.format(zscore))
                zscore_abs = np.fabs(zscore)
                result_temp = []
                temp_list = [5,10,15,20,30,50,60,80,100,150]
                print('m的取值有：{0}'.format(temp_list))
                for m in temp_list:
                    count = 0
                    index = np.argpartition(zscore_abs, -m)[-m:]
                    for each_index in index:
                        if bi_y[each_index] == 'o':
                            count += 1
                    result_temp.append(count)
                print('result_temp:{0}'.format(result_temp))

            t2 = datetime.datetime.now()
            print('从当前时间结束:{0}'.format(t2))
            print('一共用时：{0}'.format(t2 - t1))
            print('第1个数据集完毕')

        if dataset == 2:
            elem_num = 649
            filename = r"data/MF-3/data_3.dat"
            print('当前数据集是：{0}'.format(filename))
            t1 = datetime.datetime.now()
            print('从当前时间开始:{0}'.format(t1))
            X = pd.read_csv(filename, header=None, index_col=None, skiprows=0, sep=',')
            X = X.iloc[:, :649].as_matrix()
            lambda_list = [0.0001,0.001, 0.1, 1, 2,3]   #3.7
            experiment_frame(X, elem_num,lambda_list)
            # lambda_list = [2.15, 2.3, 2.45, 2.6, 2.75,
            #              3, 3.15, 3.3, 3.45, 3.6, 3.75, 4]

            lam_list = list(map(str, lambda_list))
            print(lam_list)

            y_loc = r"data/MF-3/classid_3.dat"
            y = pd.read_csv(y_loc, header=None, index_col=None, skiprows=0, sep=',')
            y = y.iloc[:, 0].values
            print(Counter(y))
            bi_y = list(map(binary_y2, y))
            print(Counter(bi_y))

            for i, lam in enumerate(lam_list):
                print("lambda:", lam)
                print('bi_y:{0}'.format(bi_y))
                print('bi_y:{0}'.format(Counter(bi_y)))
                S = np.load(folder + "\\" + "lam" + lam + "\\" + r"l21S.npk", allow_pickle=True)
                zscore = np.linalg.norm(S, axis=1)

                zscore_abs = np.fabs(zscore)
                result_temp = []
                temp_list = [20,30,50,90,100,150]
                print('m的取值有：{0}'.format(temp_list))
                for m in temp_list:
                    count = 0
                    index = np.argpartition(zscore_abs, -m)[-m:]
                    for each_index in index:
                        if bi_y[each_index] == 'o':
                            count += 1
                    result_temp.append(count)
                print('result_temp:{0}'.format(result_temp))

            t2 = datetime.datetime.now()
            print('从当前时间结束:{0}'.format(t2))
            print('一共用时：{0}'.format(t2 - t1))
            print('第2个数据集完毕')

        if dataset == 3:
            elem_num = 260
            filename = r"data/Arrhythmia_withoutdupl_05_v03.dat"
            print('当前数据集是：{0}'.format(filename))
            t1 = datetime.datetime.now()
            print('从当前时间开始:{0}'.format(t1))
            X = pd.read_csv(filename, header=None, index_col=None, skiprows=0, sep=' ')
            X = X.iloc[:, :260].values
            lambda_list = [2.2,2.25,2.3,2.35,2.4]  #2.45, 2.3
            experiment_frame(X, elem_num,lambda_list)
            # lambda_list = [2.15, 2.3, 2.45, 2.6, 2.75,
            #              3, 3.15, 3.3, 3.45, 3.6, 3.75, 4]

            lam_list = list(map(str, lambda_list))
            print(lam_list)

            y_loc = r"data/Arrhythmia_withoutdupl_05_v03.dat"
            y = pd.read_csv(y_loc, header=None, index_col=None, skiprows=0, sep=' ')
            y = y.iloc[:, 260].values
            print(Counter(y))
            bi_y = list(map(binary_y3, y))
            print(bi_y)
            print(Counter(bi_y))

            for i, lam in enumerate(lam_list):
                print("lambda:", lam)
                print('bi_y:{0}'.format(bi_y))
                print('bi_y:{0}'.format(Counter(bi_y)))
                S = np.load(folder + "\\" + "lam" + lam + "\\" + r"l21S.npk", allow_pickle=True)
                zscore = np.linalg.norm(S, axis=1)

                zscore_abs = np.fabs(zscore)
                result_temp = []
                temp_list = [5, 10, 15, 25, 30, 35, 40,45, 50, 55, 60, 80, 90, 100, 110, 120, 140, 150, 160, 170, 180, 190,
                             200]
                print('m的取值有：{0}'.format(temp_list))
                for m in temp_list:
                    count = 0
                    index = np.argpartition(zscore_abs, -m)[-m:]
                    for each_index in index:
                        if bi_y[each_index] == 'o':
                            count += 1
                    result_temp.append(count)
                print('result_temp:{0}'.format(result_temp))

            t2 = datetime.datetime.now()
            print('从当前时间结束:{0}'.format(t2))
            print('一共用时：{0}'.format(t2 - t1))
            print('第3个数据集完毕')

        if dataset == 4:
            elem_num = 649
            filename = r"data/MF-5/data_5.dat"
            print('当前数据集是：{0}'.format(filename))
            t1 = datetime.datetime.now()
            print('从当前时间开始:{0}'.format(t1))
            X = pd.read_csv(filename, header=None, index_col=None, skiprows=0, sep=',')
            X = X.iloc[:, :649].as_matrix()

            lambda_list =[0.0001,0.001, 0.1, 1, 2]
            experiment_frame(X, elem_num,lambda_list)
            lam_list = list(map(str, lambda_list))
            print(lam_list)

            y_loc = r"data/MF-5/classid_5.dat"
            y = pd.read_csv(y_loc, header=None, index_col=None, skiprows=0, sep=',')
            y = y.iloc[:, 0].values
            print(Counter(y))
            bi_y = list(map(binary_y4, y))
            print(Counter(bi_y))

            for i, lam in enumerate(lam_list):
                print("lambda:", lam)
                print('bi_y:{0}'.format(bi_y))
                print('bi_y:{0}'.format(Counter(bi_y)))
                S = np.load(folder + "\\" + "lam" + lam + "\\" + r"l21S.npk", allow_pickle=True)
                zscore = np.linalg.norm(S, axis=1)

                zscore_abs = np.fabs(zscore)
                result_temp = []
                temp_list = [20,30,50,60,70,100,150]
                print('m的取值有：{0}'.format(temp_list))
                for m in temp_list:
                    count = 0
                    index = np.argpartition(zscore_abs, -m)[-m:]
                    for each_index in index:
                        if bi_y[each_index] == 'o':
                            count += 1
                    result_temp.append(count)
                print('result_temp:{0}'.format(result_temp))

            t2 = datetime.datetime.now()
            print('从当前时间结束:{0}'.format(t2))
            print('一共用时：{0}'.format(t2 - t1))
            print('第4个数据集完毕')

        if dataset == 5:
            elem_num = 649
            filename = r"data/MF-7/data_7.dat"
            print('当前数据集是：{0}'.format(filename))
            t1 = datetime.datetime.now()
            print('从当前时间开始:{0}'.format(t1))
            X = pd.read_csv(filename, header=None, index_col=None, skiprows=0, sep=',')
            X = X.iloc[:, :649].as_matrix()

            lambda_list = [3.85,3.9,3.95]
            experiment_frame(X, elem_num, lambda_list)
            lam_list = list(map(str, lambda_list))
            print(lam_list)

            y_loc = r"data/MF-7/classid_7.dat"
            y = pd.read_csv(y_loc, header=None, index_col=None, skiprows=0, sep=',')
            y = y.iloc[:, 0].values
            print(Counter(y))
            bi_y = list(map(binary_y5, y))
            print(Counter(bi_y))

            for i, lam in enumerate(lam_list):
                print("lambda:", lam)
                print('bi_y:{0}'.format(bi_y))
                print('bi_y:{0}'.format(Counter(bi_y)))
                S = np.load(folder + "\\" + "lam" + lam + "\\" + r"l21S.npk", allow_pickle=True)
                zscore = np.linalg.norm(S, axis=1)
                print('zscore:{0}'.format(zscore))
                zscore_abs = np.fabs(zscore)
                result_temp = []
                temp_list = [20, 30, 50, 60, 90, 100, 150]
                print('m的取值有：{0}'.format(temp_list))
                for m in temp_list:
                    count = 0
                    index = np.argpartition(zscore_abs, -m)[-m:]
                    for each_index in index:
                        if bi_y[each_index] == 'o':
                            count += 1
                    result_temp.append(count)
                print('result_temp:{0}'.format(result_temp))

            t2 = datetime.datetime.now()
            print('从当前时间结束:{0}'.format(t2))
            print('一共用时：{0}'.format(t2 - t1))
            print('第5个数据集完毕')

        if dataset==6:
            elem_num=34
            filename = "data/ionosphere.txt"
            print('当前数据集是：{0}'.format(filename))
            t1 = datetime.datetime.now()
            print('从当前时间开始:{0}'.format(t1))
            X = np.loadtxt(filename, delimiter=",", usecols=np.arange(0, 34))
            lambda_list = [0.003, 0.0035, 0.004, 0.0045]  # 0.0045,0.0035
            experiment_frame(X, elem_num, lambda_list)

            lam_list = list(map(str, lambda_list))
            print(lam_list)
            y_loc = r"data/ionosphere.txt"
            y = np.loadtxt(y_loc, delimiter=",", usecols=(-1,))
            print(Counter(y))
            bi_y = list(map(binary_y6, y))
            print(Counter(bi_y))

            for i, lam in enumerate(lam_list):
                print("lambda:", lam)
                print('bi_y:{0}'.format(bi_y))
                print('bi_y:{0}'.format(Counter(bi_y)))
                S = np.load(folder + "\\" + "lam" + lam + "\\" + r"l21S.npk", allow_pickle=True)
                zscore = np.linalg.norm(S, axis=1)

                zscore_abs = np.fabs(zscore)
                result_temp = []
                temp_list = [5, 10, 30, 60, 90, 120, 130, 140, 150, 200, 300, 340]
                print('m的取值有：{0}'.format(temp_list))
                for m in temp_list:
                    count = 0
                    index = np.argpartition(zscore_abs, -m)[-m:]
                    for each_index in index:
                        if bi_y[each_index] == 'o':
                            count += 1
                    result_temp.append(count)
                print('result_temp:{0}'.format(result_temp))

            t2 = datetime.datetime.now()
            print('从当前时间结束:{0}'.format(t2))
            print('一共用时：{0}'.format(t2 - t1))
            print('第5个数据集完毕')

        if dataset==7:
            elem_num=166
            filename = r"data/clean2.data"
            print('当前数据集是：{0}'.format(filename))
            t1 = datetime.datetime.now()
            print('从当前时间开始:{0}'.format(t1))

            X = pd.read_csv(filename, header=None, index_col=None, skiprows=0, sep=',')
            X = X.iloc[:, 2:168].values
            lambda_list = [0.24, 0.25, 0.255, 0.26, 0.265]
            experiment_frame(X, elem_num, lambda_list)

            lam_list = list(map(str, lambda_list))
            print(lam_list)

            y_loc = r"data/clean2.data"
            y = pd.read_csv(y_loc, header=None, index_col=None, skiprows=0, sep=',')
            y = y.iloc[:, 168].values
            print(Counter(y))
            bi_y = list(map(binary_y3, y))
            print(Counter(bi_y))

            for i, lam in enumerate(lam_list):
                print("lambda:", lam)
                print('bi_y:{0}'.format(bi_y))
                print('bi_y:{0}'.format(Counter(bi_y)))
                S = np.load(folder + "\\" + "lam" + lam + "\\" + r"l21S.npk", allow_pickle=True)
                zscore = np.linalg.norm(S, axis=1)

                zscore_abs = np.fabs(zscore)
                result_temp = []
                temp_list = [1000, 2000, 3000, 4000, 5000, 6000, 6598]
                print('m的取值有：{0}'.format(temp_list))
                for m in temp_list:
                    count = 0
                    index = np.argpartition(zscore_abs, -m)[-m:]
                    for each_index in index:
                        if bi_y[each_index] == 'o':
                            count += 1
                    result_temp.append(count)
                print('result_temp:{0}'.format(result_temp))

            t2 = datetime.datetime.now()
            print('从当前时间结束:{0}'.format(t2))
            print('一共用时：{0}'.format(t2 - t1))
            print('第5个数据集完毕')


