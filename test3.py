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


def binary_error(value):
    if value == 0.0:
        return "m"  # 'majority'
    else:
        return "o"  # 'outlier'


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

if __name__ == "__main__":

    folder = 'OutlierDetectionResult'
    for n in range(1,4):
        dataset = 3
        if dataset==1:
            elem_num=617
            filename=r"data/ISOLET-23/data_23.dat"
            print(filename)
            X = pd.read_csv(filename, header=None, index_col=None, skiprows=0, sep=',')
            X = X.iloc[:,:617].values

            lambda_list = [0.0015,0.0016,0.0017,0.0018]   #0.0017,0.0018  ,0.0015
            experiment_frame(X,elem_num,lambda_list)

            lam_list = list(map(str, lambda_list))
            print(lam_list)

            y_loc = r"data/ISOLET-23/classid_23.dat"
            y = pd.read_csv(y_loc, header=None, index_col=None, skiprows=0, sep=',')
            y = y.iloc[:,0].values
            print(Counter(y))
            bi_y = list(map(binary_y, y))
            print(Counter(bi_y))

            precisions = []
            lams = []
            recalls = []
            f1s = []
            for i, lam in enumerate(lam_list):
                S = np.load(folder + "\\" + "lam" + lam + "\\" + r"l21S.npk", allow_pickle=True)
                predictions = list(map(binary_error, np.linalg.norm(S, axis=1)))
                print("lambda:", lam)
                print('bi_y:{0}'.format(bi_y))
                print('bi_y:{0}'.format(Counter(bi_y)))
                print('predictions:{0}'.format(predictions))
                print('predictions:{0}'.format(Counter(predictions)))

                result_temp = []
                temp_list = [5,10,15,20,30,50,60,80,100,150]
                max_pred = Counter(predictions)['o']
                print('max_pre:{0}'.format(max_pred))
                for m in temp_list:
                    m_count = 0
                    real_count = 0
                    if m > max_pred:
                        m = max_pred
                    for index, j in enumerate(predictions):
                        if m_count < m:
                            if j == 'o':
                                m_count += 1
                                if bi_y[index] == 'o':
                                    real_count += 1
                        else:
                            result_temp.append(real_count)
                            break
                print(result_temp)

                print("precision", precision(bi_y, predictions, labels=["o", "m"], pos_label="o"))
                print("recall", recall(bi_y, predictions, labels=["o", "m"], pos_label="o"))
                print("f1", f1_score(bi_y, predictions, labels=["o", "m"], pos_label="o"))
                lams.append(lam)
                precisions.append(precision(bi_y, predictions, labels=["o", "m"], pos_label="o"))
                recalls.append(recall(bi_y, predictions, labels=["o", "m"], pos_label="o"))
                f1s.append(f1_score(bi_y, predictions, labels=["o", "m"], pos_label="o"))
                print(CM(bi_y, predictions))
                print("------------")
            print(len(lams), len(recalls), len(f1s), len(precisions))

            d = {"lambda": list(map(float, lams)), "precision": precisions, "recall": recalls, "f1": f1s}
            data = pd.DataFrame(d)
            print(data)
            result = data.sort_values(by=["lambda"], ascending=True)
            print(result)

            l = list(range(len(lams)))
            plt.figure(figsize=(6.5, 4.5))
            plt.xlabel("Lambdas")
            plt.ylabel("Values")
            plt.plot(l, result.f1, color='r', label="f1")
            plt.plot(l, result.precision, color="b", label="precision")
            plt.plot(l, result.recall, color="g", label="recall")
            plt.legend(["f1", "precision", "recall"], loc='best')
            plt.xticks(l, result["lambda"], rotation='vertical')
            plt.title("Anomalies Detection of $l_{2,1}$ Robust Auto-encoder")
            plt.show()
            print('第1个数据集完毕')

        if dataset == 2:
            elem_num = 650
            filename = r"data/MF-3/data_3.dat"
            print(filename)
            X = pd.read_csv(filename, header=None, index_col=None, skiprows=0, sep=',')
            X = X.values
            lambda_list = [3.6,3.7,3.8,3.9]   #3.7
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

            precisions = []
            lams = []
            recalls = []
            f1s = []
            for i, lam in enumerate(lam_list):
                S = np.load(folder + "\\" + "lam" + lam + "\\" + r"l21S.npk", allow_pickle=True)
                print(S)
                predictions = list(map(binary_error, np.linalg.norm(S, axis=1)))

                print('bi_y:{0}'.format(bi_y))
                print('bi_y:{0}'.format(Counter(bi_y)))
                print('predictions:{0}'.format(predictions))
                print('predictions:{0}'.format(Counter(predictions)))
                result_temp = []
                temp_list = [20,30,50,60,90,100,150]
                max_pred = Counter(predictions)['o']
                print('max_pre:{0}'.format(max_pred))
                for m in temp_list:
                    m_count = 0
                    real_count = 0
                    if m > max_pred:
                        m = max_pred
                    for index, j in enumerate(predictions):
                        if m_count < m:
                            if j == 'o':
                                m_count += 1
                                if bi_y[index] == 'o':
                                    real_count += 1
                        else:
                            result_temp.append(real_count)
                            break
                print(result_temp)

                print("precision", precision(bi_y, predictions, labels=["o", "m"], pos_label="o"))
                print("recall", recall(bi_y, predictions, labels=["o", "m"], pos_label="o"))
                print("f1", f1_score(bi_y, predictions, labels=["o", "m"], pos_label="o"))
                lams.append(lam)
                precisions.append(precision(bi_y, predictions, labels=["o", "m"], pos_label="o"))
                recalls.append(recall(bi_y, predictions, labels=["o", "m"], pos_label="o"))
                f1s.append(f1_score(bi_y, predictions, labels=["o", "m"], pos_label="o"))
                print(CM(bi_y, predictions))
                print("------------")
            print(len(lams), len(recalls), len(f1s), len(precisions))

            d = {"lambda": list(map(float, lams)), "precision": precisions, "recall": recalls, "f1": f1s}
            data = pd.DataFrame(d)
            result = data.sort_values(by=["lambda"], ascending=True)

            l = list(range(len(lams)))
            plt.figure(figsize=(6.5, 4.5))
            plt.xlabel("Lambdas")
            plt.ylabel("Values")
            plt.plot(l, result.f1, color='r', label="f1")
            plt.plot(l, result.precision, color="b", label="precision")
            plt.plot(l, result.recall, color="g", label="recall")
            plt.legend(["f1", "precision", "recall"], loc='best')
            plt.xticks(l, result["lambda"], rotation='vertical')
            plt.title("Anomalies Detection of $l_{2,1}$ Robust Auto-encoder")
            plt.show()
            print('第2个数据集完毕')

        if dataset == 3:
            elem_num = 260
            filename = r"data/Arrhythmia_withoutdupl_05_v03.dat"
            print(filename)
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

            precisions = []
            lams = []
            recalls = []
            f1s = []
            for i, lam in enumerate(lam_list):
                S = np.load(folder + "\\" + "lam" + lam + "\\" + r"l21S.npk", allow_pickle=True)
                print(S)
                predictions = list(map(binary_error, np.linalg.norm(S, axis=1)))
                print("lambda:", lam)
                print('bi_y:{0}'.format(bi_y))
                print('bi_y:{0}'.format(Counter(bi_y)))
                print('predictions:{0}'.format(predictions))
                print('predictions:{0}'.format(Counter(predictions)))
                result_temp = []
                temp_list = [5, 10, 15, 25, 30, 35, 45, 50, 55, 60, 80, 90, 100, 110, 120, 140, 150, 160, 170, 180, 190,
                             200]
                max_pred = Counter(predictions)['o']
                print('max_pre:{0}'.format(max_pred))
                for m in temp_list:
                    m_count = 0
                    real_count = 0
                    if m > max_pred:
                        m = max_pred
                    for index, j in enumerate(predictions):
                        if m_count < m:
                            if j == 'o':
                                m_count += 1
                                if bi_y[index] == 'o':
                                    real_count += 1
                        else:
                            result_temp.append(real_count)
                            break
                print(result_temp)

                print("lambda:", lam)
                print("precision", precision(bi_y, predictions, labels=["o", "m"], pos_label="o"))
                print("recall", recall(bi_y, predictions, labels=["o", "m"], pos_label="o"))
                print("f1", f1_score(bi_y, predictions, labels=["o", "m"], pos_label="o"))
                lams.append(lam)
                precisions.append(precision(bi_y, predictions, labels=["o", "m"], pos_label="o"))
                recalls.append(recall(bi_y, predictions, labels=["o", "m"], pos_label="o"))
                f1s.append(f1_score(bi_y, predictions, labels=["o", "m"], pos_label="o"))
                print(CM(bi_y, predictions))
                print("------------")
            print(len(lams), len(recalls), len(f1s), len(precisions))

            d = {"lambda": list(map(float, lams)), "precision": precisions, "recall": recalls, "f1": f1s}
            data = pd.DataFrame(d)
            result = data.sort_values(by=["lambda"], ascending=True)
            l = list(range(len(lams)))
            plt.figure(figsize=(6.5, 4.5))
            plt.xlabel("Lambdas")
            plt.ylabel("Values")
            plt.plot(l, result.f1, color='r', label="f1")
            plt.plot(l, result.precision, color="b", label="precision")
            plt.plot(l, result.recall, color="g", label="recall")
            plt.legend(["f1", "precision", "recall"], loc='best')
            plt.xticks(l, result["lambda"], rotation='vertical')
            plt.title("Anomalies Detection of $l_{2,1}$ Robust Auto-encoder")
            plt.show()
            print('第3个数据集完毕')


