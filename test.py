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


def experiment_frame(_file_name):
    X=np.loadtxt(_file_name, delimiter=",",usecols=np.arange(0,34))
    # X = np.load(r"data/data.npk", allow_pickle=True)
    inner = 100
    outer = 8

    lamda_list = [0.003,0.0035,0.004,0.0045]
    layers = [34, 400, 200]  ## S trans
    folder = r"OutlierDetectionResult"
    if not os.path.isdir(folder):
        os.makedirs(folder)
    os.chdir(folder)

    for lam in lamda_list:
        folder = "lam" + str(lam)
        l21RDAE(X=X, layers=layers, lamda=lam, folder=folder, learning_rate=0.005,
                inner=inner, outer=outer, batch_size=133, inputsize=(28,28))
    os.chdir("../")

def binary_y(value):
    if value == 1:
        return "m"
    else:
        return "o"

if __name__ == "__main__":
    folder = 'OutlierDetectionResult'
    file_name="data/ionosphere.txt"
    # experiment_frame(file_name)
    lambda_list = [0.003,0.0035,0.004,0.0045]   #0.0045,0.0035
    lam_list = list(map(str, lambda_list))
    print(lam_list)

    y_loc = r"data/ionosphere.txt"
    y = np.loadtxt(y_loc, delimiter=",",usecols=(-1,))
    print(Counter(y))
    bi_y = list(map(binary_y, y))
    print(Counter(bi_y))

    precisions = []
    lams = []
    recalls = []
    f1s = []
    for i, lam in enumerate(lam_list):
        print("lambda:", lam)
        print('bi_y:{0}'.format(bi_y))
        print('bi_y:{0}'.format(Counter(bi_y)))
        S = np.load(folder + "\\" + "lam" + lam+ "\\" + r"l21S.npk", allow_pickle=True)
        zscore=np.linalg.norm(S, axis=1)

        zscore_abs = np.fabs(zscore)
        result_temp = []
        temp_list = [5,10,30,60,90,120,130,140,150,200,300,340]
        print('m的取值有：{0}'.format(temp_list))
        for m in temp_list:
            count = 0
            index = np.argpartition(zscore_abs, -m)[-m:]
            for each_index in index:
                if bi_y[each_index] == 'o':
                    count += 1
            result_temp.append(count)
        print('result_temp:{0}'.format(result_temp))




