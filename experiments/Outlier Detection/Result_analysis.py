import numpy as np
import pandas as pd
import os

from imageio import imread as im
import matplotlib.pyplot as plt

# from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import confusion_matrix as CM
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision
from sklearn.decomposition import PCA

from collections import Counter

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM



def binary_error(value):
    if value == 0.0:
        return "m"  # 'majority'
    else:
        return "o"  # 'outlier'


def binary_y(value):
    if value == 4:
        return "m"
    else:
        return "o"

folder = os.getcwd()
lambda_list = [0.0001, 0.0003, 0.0008, 0.001, 0.0015, 0.00035, 0.00045,
         0.00055, 0.00065, 0.00075, 0.00085, 0.00095, 0.00105, 0.00115, 0.00125]

lam_list = list(map(str,lambda_list))
print (lam_list)

y_loc = r"..\..\data\y.npk"
y = np.load(y_loc,allow_pickle=True)
print (Counter(y))
print (len(y) - Counter(y)[4])

bi_y = list(map(binary_y,y))
print (Counter(bi_y))


precisions=[]
lams=[]
recalls=[]
f1s = []
for i,lam in enumerate(lam_list):
    S = np.load(folder + "\\" + "lam" + lam + "\\" + r"l21S.npk",allow_pickle=True)
    predictions = list(map(binary_error,np.linalg.norm(S,axis = 1)))
    print ("lambda:", lam)
    print ("precision",precision(bi_y,predictions,labels=["o","m"],pos_label="o"))
    print ("recall",recall(bi_y,predictions,labels=["o","m"],pos_label="o"))
    print ("f1",f1_score(bi_y,predictions,labels=["o","m"],pos_label="o"))
    lams.append(lam)
    precisions.append(precision(bi_y,predictions,labels=["o","m"],pos_label="o"))
    recalls.append(recall(bi_y,predictions,labels=["o","m"],pos_label="o"))
    f1s.append(f1_score(bi_y,predictions,labels=["o","m"],pos_label="o"))
    print (CM(bi_y,predictions))
    print ("------------")
print (len(lams),len(recalls),len(f1s),len(precisions))


d = {"lambda":list(map(float,lams)),"precision":precisions,"recall":recalls,"f1":f1s}
data = pd.DataFrame(d)
print (data)
result = data.sort_values(by=["lambda"],ascending=True)
print (result)

l = list(range(len(lams)))
plt.figure(figsize=(6.5,4.5))
plt.xlabel("Lambdas")
plt.ylabel("Values")
plt.plot(l,result.f1,color='r',label="f1")
plt.plot(l,result.precision,color="b",label="precision")
plt.plot(l,result.recall,color="g",label="recall")
plt.legend(["f1","precision","recall"],loc='best')
plt.xticks(l, result["lambda"],rotation='vertical')
plt.title("Anomalies Detection of $l_{2,1}$ Robust Auto-encoder")
plt.show()

ncol = 3
folder_cor = folder + "\\"  ##+corruption_level_folder[-2]
fig, ax = plt.subplots(nrows=len(lam_list), ncols=ncol)
X = im(folder_cor + "\\" + r"X.png")
for index in range(len(lam_list)):
    # rR = im(folder_cor + "\\" + "lam" +lam_list[index]+ "\\" + r"rR.png")
    l21R = im(folder_cor + "\\" + "lam" + lam_list[index] + "\\" + r"l21R.png")
    l21S = im(folder_cor + "\\" + "lam" + lam_list[index] + "\\" + r"l21S.png")
    # rS = im(folder_cor + "\\" + "lam" +lam_list[index] + "\\" + r"rS.png")

    ax[index][0].imshow(l21R, cmap="gray")
    ax[index][1].imshow(l21S, cmap="gray")
    ax[index][2].imshow(X, cmap="gray")

    ax[index][0].set_title(lam_list[index] + r" L21 R")
    ax[index][1].set_title(lam_list[index] + r" L21 S")
    ax[index][2].set_title("X")
    ax[index][0].get_xaxis().set_visible(False)
    ax[index][0].get_yaxis().set_visible(False)
    ax[index][1].get_xaxis().set_visible(False)
    ax[index][1].get_yaxis().set_visible(False)
    ax[index][2].get_xaxis().set_visible(False)
    ax[index][2].get_yaxis().set_visible(False)

fig.set_size_inches(16,(len(lam_list))*3.5)
#fig.savefig(r"C:\Users\zc\Desktop\Result\Comparing.png",bbox_inches='tight')
plt.show()


def binary_y(value):
    if value == 4:
        return 1
    else:
        return -1

y_loc = r"..\..\data\y.npk"
x_loc = r"..\..\data\data.npk"

y = np.load(y_loc,allow_pickle=True)
x = np.load(x_loc,allow_pickle=True)
print (Counter(y))
print (len(y) - Counter(y)[4])
print (x.shape)

fractions = np.arange(0.01,0.7,0.02)
y_preds = []
for fraction in fractions:
    model = IsolationForest(n_estimators=100,contamination=fraction)
    model.fit(x)
    y_pred = model.predict(x)
    y_preds.append(y_pred)

for i in y_preds:
    print (Counter(i))

precisions=[]
fraction_list=[]
recalls=[]
f1s = []

bi_y = list(map(binary_y,y))
for fraction,predictions in zip(fractions,y_preds):
    print ("fraction",fraction)
    print ("precision",precision(bi_y,predictions,labels=[1,-1],pos_label=-1))
    print ("recall",recall(bi_y,predictions,labels=[1,-1],pos_label=-1))
    print ("f1",f1_score(bi_y,predictions,labels=[1,-1],pos_label=-1))
    fraction_list.append(fraction)
    precisions.append(precision(bi_y,predictions,labels=[1,-1],pos_label=-1))
    recalls.append(recall(bi_y,predictions,labels=[1,-1],pos_label=-1))
    f1s.append(f1_score(bi_y,predictions,labels=[1,-1],pos_label=-1))

    print (CM(bi_y,predictions))
    print ("------------")


plt.figure(figsize=(6.5,4.5))
plt.xlabel("fractions")
plt.ylabel("Values")
plt.plot(range(len(fraction_list)),f1s,color='r',label="f1")
plt.plot(range(len(fraction_list)),precisions,color="b",label="precision")
plt.plot(range(len(fraction_list)),recalls,color="g",label="recall")
plt.legend(["f1","precision","recall"],loc='best')
plt.xticks(range(len(fraction_list)), fraction_list,rotation='vertical')
plt.title("Anomalies Detection of Isolation Forest")
plt.show()


fractions = np.arange(0.01,0.7,0.02)
y_preds = []
for fraction in fractions:
    model = OneClassSVM(nu=fraction)
    model.fit(x)
    y_pred = model.predict(x)
    y_preds.append(y_pred)



precisions=[]
fraction_list=[]
recalls=[]
f1s = []


bi_y = list(map(binary_y,y))
for fraction,predictions in zip(fractions,y_preds):
    print ("fraction",fraction)
    print ("precision",precision(bi_y,predictions,labels=[1,-1],pos_label=-1))
    print ("recall",recall(bi_y,predictions,labels=[1,-1],pos_label=-1))
    print ("f1",f1_score(bi_y,predictions,labels=[1,-1],pos_label=-1))
    fraction_list.append(fraction)
    precisions.append(precision(bi_y,predictions,labels=[1,-1],pos_label=-1))
    recalls.append(recall(bi_y,predictions,labels=[1,-1],pos_label=-1))
    f1s.append(f1_score(bi_y,predictions,labels=[1,-1],pos_label=-1))

    print (CM(bi_y,predictions))
    print ("------------")


plt.figure(figsize=(6.5,4.5))
plt.xlabel("Fractions")
plt.ylabel("Scores")
plt.plot(range(len(fraction_list)),f1s,color='r',label="f1")
plt.plot(range(len(fraction_list)),precisions,color="b",label="precision")
plt.plot(range(len(fraction_list)),recalls,color="g",label="recall")
plt.legend(["f1","precision","recall"],loc='best')
plt.xticks(range(len(fraction_list)), fraction_list,rotation='vertical')
plt.show()
