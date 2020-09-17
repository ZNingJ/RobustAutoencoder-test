import PIL.Image as Image
import ImShow as I
import numpy as np
import tensorflow as tf
import l21RobustAutoencoderOnST as l21RDAE
import os

def l21RDAE_compressFeature(X, layers, lamda, folder, learning_rate = 0.01, inner = 100, outer = 10, batch_size = 133,inputsize = (28,28)):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    os.chdir(folder)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            rael21 = l21RDAE.RobustL21Autoencoder(sess = sess, lambda_= lamda*X.shape[0], layers_sizes=layers)
            l21L, l21S = rael21.fit(X = X, sess = sess, inner_iteration = inner, iteration = outer, batch_size = batch_size, learning_rate = learning_rate,  verbose = True)
            l21R = rael21.getRecon(X = X, sess = sess)
            l21H = rael21.transform(X, sess)
            l21H.dump(r"l21H.pkl")
            l21S.dump("l21S.pkl")
    os.chdir("../")

def compare_frame():

    X = np.load(r"/home/zc8304/Documents/packets_1000.npk")
    X = X[:,0:1512] ## truncate the matrix if neccessary
    inner = 120
    outer = 10

    lamda_list = np.arange(0.00005,0.001,0.00005)


    layers = [1512, 400, 200]

    for lam in lamda_list:
        folder = "lam" + str(lam)

        l21RDAE_compressFeature(X = X, layers=layers, lamda = lam, folder = folder, learning_rate = 0.05, inner = inner, outer = outer, batch_size = 133,inputsize = (63,72))
if __name__ == "__main__":
    compare_frame()
