#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 22:47:25 2018

@author: mson
"""

import sys
sys.path.append('..')
import numpy as np
from sklearn.model_selection import train_test_split
import neural_network.activations as avs

def trn_val_tst(X, y, trn_prop, val_prop, tst_prop):
    assert(0.99 <= trn_prop + val_prop + tst_prop <= 1.01)    
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y,
                                                  test_size = val_prop + tst_prop,
                                                  random_state = 1)
    X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn,
                                                  test_size = tst_prop,
                                                  random_state = 1)
    return X_trn, y_trn, X_val, y_val, X_tst, y_tst

def standardize_cols(X):
    return (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)

def bind_and_sort(y, yhat):
    """ Put input vectors into a matrix and sort by yhat descending"""
    yyhat = np.vstack((y, yhat)).T
    return yyhat[yyhat[:,1].argsort()[::-1]]

def standard_binary_classification_layers(n_hidden_layers):
    """ return a list of n activation functions for a net of n hidden layers:
    all reLU activations except for the last, which is sigmoid """
    activation_functions = [avs.relu for i in range(n_hidden_layers - 1)]
    activation_functions.append(avs.sigmoid)
    return activation_functions

class Standardizer:
    def __init__(self, X_train):
        self.mean = np.mean(X_train, axis = 0)
        self.sd = np.std(X_train, axis = 0)

    def standardize(self, X):
        return (X - self.mean) / self.sd