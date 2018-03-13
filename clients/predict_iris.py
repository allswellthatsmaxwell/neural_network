#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 16:58:57 2018

@author: mson
"""

import sys
sys.path.append('..')

import numpy as np
import neural_network.neural_network as nn
import neural_network.loss_functions as losses
from neural_network.activations import relu, sigmoid 
from sklearn import datasets
from sklearn.metrics import roc_auc_score
from prediction_utils import trn_val_tst
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X, y = iris.data, iris.target
y_binary = np.array([1 if el in (1, 2) else 0 for el in y])
X_trn, y_trn, X_val, y_val, X_tst, y_tst = trn_val_tst(X, y_binary, 
                                                       4/10, 3/10, 3/10)
ilayer_dims = [X.shape[1], 4, 1]
iris_net = nn.Net(ilayer_dims, [relu, relu, sigmoid], loss = losses.LogLoss(),
                  use_adam = True)

costs = iris_net.train(X_trn.T, y_trn, iterations = 200, learning_rate = 0.1,                
                       beta1 = 0.9, beta2 = 0.99, lambd = 0, 
                       minibatch_size = X_trn.shape[0],
                       debug = True)
yhat_trn = iris_net.predict(X_trn.T)
yhat_val = iris_net.predict(X_val.T)
yyhat_trn = np.vstack((y_trn, yhat_trn)).T
yyhat_val = np.vstack((y_val, yhat_val)).T

auc_val = roc_auc_score(y_val, yhat_val)
print("AUC =", auc_val)