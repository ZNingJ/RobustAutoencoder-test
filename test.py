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

    lamda_list = [0.0023,0.0025,0.0028]
    layers = [34, 400, 200]  ## S trans
    # layers = [784, 400, 200]  ## S trans
    folder = r"OutlierDetectionResult"
    if not os.path.isdir(folder):
        os.makedirs(folder)
    os.chdir(folder)

    for lam in lamda_list:
        folder = "lam" + str(lam)
        l21RDAE(X=X, layers=layers, lamda=lam, folder=folder, learning_rate=0.005,
                inner=inner, outer=outer, batch_size=133, inputsize=(28,28))
    os.chdir("../")


def binary_error(value):
    if value == 0.0:
        return "m"  # 'majority'
    else:
        return "o"  # 'outlier'


def binary_y(value):
    if value == 1:
        return "m"
    else:
        return "o"

if __name__ == "__main__":
    folder = 'OutlierDetectionResult'
    file_name="data/ionosphere.txt"
    experiment_frame(file_name)
    lambda_list = [0.0023,0.0025,0.0028]   #0.0045,0.0035
    lam_list = list(map(str, lambda_list))
    print(lam_list)

    y_loc = r"data/ionosphere.txt"
    y = np.loadtxt(y_loc, delimiter=",",usecols=(-1,))
    print(Counter(y))
    print(len(y) - Counter(y)[1])
    bi_y = list(map(binary_y, y))
    print(Counter(bi_y))

    precisions = []
    lams = []
    recalls = []
    f1s = []
    for i, lam in enumerate(lam_list):
        S = np.load(folder + "\\" + "lam" + lam+ "\\" + r"l21S.npk", allow_pickle=True)
        print(S)
        predictions = list(map(binary_error, np.linalg.norm(S, axis=1)))
        print("lambda:", lam)
        print('bi_y:{0}'.format(bi_y))
        print('bi_y:{0}'.format(Counter(bi_y)))
        print('predictions:{0}'.format(predictions))
        print('predictions:{0}'.format(Counter(predictions)))

        result_temp = []
        temp_list = [5, 10, 30, 60, 90, 120, 130, 140, 150, 200, 300, 340]
        max_pred = Counter(predictions)['o']
        for m in temp_list:
            m_count = 0
            real_count = 0
            s=m
            if m > max_pred:
                m = max_pred
            for index, j in enumerate(predictions):
                if m_count < m:
                    if j == 'o':
                        m_count += 1
                        if bi_y[index] == 'o':
                            real_count += 1
                    if index==len(predictions)-1:
                        result_temp.append(real_count)
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


