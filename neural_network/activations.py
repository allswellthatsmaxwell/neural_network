#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 20:23:11 2018

@author: Maxwell Peterson

Activation functions and their backward-propogation companions.
"""

import numpy as np

def sigmoid(Z):
    Z = np.clip(Z, -400, 400) ## prevent overflow
    return 1 / (1 + np.exp(-Z))

def relu(Z):
    return np.maximum(0, Z)        

def softmax(Z):
    """Compute the softmax of vector Z in a numerically stable way."""
    shiftZ = Z - np.max(Z)
    exps = np.exp(shiftZ)
    return exps / np.sum(exps)

def relu_prime(dA, Z):
    """
    Perform backward propagation for a single RELU unit.    
    Arguments: dA -- post-activation gradient, of any shape
    Returns: dZ -- Gradient of the cost with respect to Z
    """
    dZ = np.array(dA, copy = True) # just converting dz to a correct object.
    assert(not dZ is dA)
    assert (dZ.shape == Z.shape)
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    return dZ

def sigmoid_prime(dA, Z):
    """
    Perform backward propagation for a single SIGMOID unit.
    Arguments: dA -- post-activation gradient, of any shape
    Returns: dZ -- Gradient of the cost with respect to Z
    """
    s = sigmoid(Z)
    dZ = dA * s * (1 - s)    
    assert (dZ.shape == Z.shape)    
    return dZ

backward_map = {relu: relu_prime, sigmoid: sigmoid_prime}

def derivative(activation):
    return backward_map[activation]