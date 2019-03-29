#!/usr/bin/env python3

import numpy as np
import struct
import os
import gzip
import urllib.request

def main():
    X_train, y_train, X_test, y_test = get_MNIST()

    #Theta = softmax_gd(X_train, y_train, X_test, y_test, epochs=10, alpha=0.5)
    Theta = softmax_sgd(X_train, y_train, X_test, y_test, epochs=10, alpha=0.01)

def get_MNIST():
    fnames = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte",
              "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]
    for fname in fnames:
        download(fname)
    X_train = parse_images("train-images-idx3-ubyte")
    y_train = parse_labels("train-labels-idx1-ubyte")
    X_test = parse_images("t10k-images-idx3-ubyte")
    y_test = parse_labels("t10k-labels-idx1-ubyte")
    return X_train, y_train, X_test, y_test

def download(fname):
    if not os.path.exists(fname):
        print("Downloading: {}".format(fname))
        url = 'http://yann.lecun.com/exdb/mnist/' + fname + '.gz'
        with urllib.request.urlopen(url) as response, open(fname, 'wb') as f:
            f.write(gzip.decompress(response.read()))

def parse_images(filename):
    f = open(filename,"rb");
    magic,size = struct.unpack('>ii', f.read(8))
    sx,sy = struct.unpack('>ii', f.read(8))
    X = []
    for i in range(size):
        im =  struct.unpack('B'*(sx*sy), f.read(sx*sy))
        X.append([float(x)/255.0 for x in im]);
    return np.array(X);

def parse_labels(filename):
    one_hot = lambda x, K: np.array(x[:,None] == np.arange(K)[None, :],
                                    dtype=np.float64)
    f = open(filename,"rb");
    magic,size = struct.unpack('>ii', f.read(8))
    return one_hot(np.array(struct.unpack('B'*size, f.read(size))), 10)

def error(y_hat,y):
    return float(np.sum(np.argmax(y_hat,axis=1) !=
                        np.argmax(y,axis=1)))/y.shape[0]


# helper functions for loss
softmax_loss = lambda yp,y : (np.log(np.sum(np.exp(yp))) - yp.dot(y),
                              np.exp(yp)/np.sum(np.exp(yp)) - y)

##### Implement the functions below this point ######

def softmax_gd(X, y, Xt, yt, epochs=10, alpha = 0.5):
    """
    Run gradient descent to solve linear softmax regression.

    Inputs:
        X: numpy array of training inputs
        y: numpy array of training outputs
        Xt: numpy array of testing inputs
        yt: numpy array of testing outputs
        epochs: number of passes to make over the whole training set
        alpha: step size

    Outputs:
        Theta: 10 x 785 numpy array of trained weights
    """
    Theta = np.zeros((y.shape[1],X.shape[1]+1))
    # Your implementation here.
    col_ones = np.ones((X.shape[0], 1))
    X = np.concatenate((X, col_ones), axis=1)

    col_ones = np.ones((Xt.shape[0], 1))
    Xt = np.concatenate((Xt, col_ones), axis=1)

    for e in range(epochs):
        print('##### '+str(e)+' #########')
        g = np.zeros((y.shape[1],X.shape[1]))
        softmaxLossValues = []
        for i in range(X.shape[0]):

            loss_value, loss_grad = softmax_loss(np.dot(Theta, X[i]), y[i])
            softmaxLossValues.append(loss_value)
            #print('a: ', loss_grad.shape) #10x1
            #print('b: ', np.transpose(newX).shape) #need it to be 1x785
            g = g + (1/X.shape[0])*np.outer(loss_grad, X[i])
        print('train error: ', error(np.matmul(X, np.transpose(Theta)), y))
        print('avg train softmax lost this epoch: ', sum(softmaxLossValues)/len(softmaxLossValues))
        
        #test the model
        test_softmax_gd(Theta, Xt, yt)

        Theta = Theta - alpha * g
    return Theta

#helper for evaluating test set for softmaxGD
def test_softmax_gd(Theta, Xt, yt):
    softmaxLossValues2 = []
    #g = np.zeros((yt.shape[1],Xt.shape[1]))
    for i in range(Xt.shape[0]):
        loss_value, loss_grad = softmax_loss(np.dot(Theta, Xt[i]), yt[i])
        softmaxLossValues2.append(loss_value)
        #print('a: ', loss_grad.shape) #10x1
        #print('b: ', np.transpose(newX).shape) #need it to be 1x785
        #g = g + (1/Xt.shape[0])*np.outer(loss_grad, Xt[i])
    print('test error: ', error(np.matmul(Xt, np.transpose(Theta)), yt))
    print('avg test softmax lost this epoch: ', sum(softmaxLossValues2)/len(softmaxLossValues2))


def softmax_sgd(X,y, Xt, yt, epochs=10, alpha = 0.01):
    """
    Run stochastic gradient descent to solve linear softmax regression.

    Inputs:
        X: numpy array of training inputs
        y: numpy array of training outputs
        Xt: numpy array of testing inputs
        yt: numpy array of testing outputs
        epochs: number of passes to make over the whole training set
        alpha: step size

    Outputs:
        Theta: 10 x 785 numpy array of trained weights
    """
    Theta = np.zeros((y.shape[1],X.shape[1]+1))
    # Your implementation here.
    col_ones = np.ones((X.shape[0], 1))
    X = np.concatenate((X, col_ones), axis=1)

    col_ones = np.ones((Xt.shape[0], 1))
    Xt = np.concatenate((Xt, col_ones), axis=1)
    for e in range(epochs):
        print('-------- epoch ' + str(e) + '-------------')
        softmaxLossValues = []
        testSoftmaxLossValues = []
        print('train error: ', error(np.matmul(X, np.transpose(Theta)), y))
        print('test error: ', error(np.matmul(Xt, np.transpose(Theta)), yt))
        if e > 0:
            print('avg train softmax lost this epoch: ', avgTrainLoss)
            print('avg test softmax loss this epoch: ', avgTestLoss)
        else: #how to get the losses on zeroth epoch? need to iterate without updating Theta
            firstTrainLosses = []
            for i in range(X.shape[0]):
                loss_value, loss_grad = softmax_loss(np.dot(Theta, X[i]), y[i])
                firstTrainLosses.append(loss_value)
            print('avg train softmax lost this epoch: ', sum(firstTrainLosses)/len(firstTrainLosses))

            ####test set########
            firstTestLosses = []
            for i in range(Xt.shape[0]):
                testloss, testlossgrad = softmax_loss(np.dot(Theta, Xt[i]), yt[i])
                firstTestLosses.append(testloss)
            print('avg test softmax loss this epoch: ', sum(firstTestLosses)/len(firstTestLosses))
            #########################################################


        #iterate over examples and update Theta
        for i in range(X.shape[0]):
            loss_value, loss_grad = softmax_loss(np.dot(Theta, X[i]), y[i])
            softmaxLossValues.append(loss_value)

            Theta = Theta - alpha*np.outer(loss_grad, X[i])
        avgTrainLoss = sum(softmaxLossValues)/len(softmaxLossValues)

        #######test set performance
        for i in range(Xt.shape[0]):
            testLoss, testGrad = softmax_loss(np.dot(Theta, Xt[i]), yt[i])
            testSoftmaxLossValues.append(testLoss)
        avgTestLoss = sum(testSoftmaxLossValues)/len(testSoftmaxLossValues)

    return Theta

if __name__=='__main__':
    main()