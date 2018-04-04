#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 23:42:03 2018

@author: Maxwell Peterson
"""

import neural_network.activations as actv
import neural_network.initializations
import numpy as np
import neural_network.loss_functions
import warnings

class InputLayer:
    
    def __init__(self, A):
        self.A = A
        self.name = "input"
        self.m_examples = A.shape[1]

class Layer:

    def __init__(self, name, n, n_prev, activation,
                 use_adam = False,
                 initialization = neural_network.initializations.he,
                 keep_prob = 1):
        """ n: integer; the dimension of this layer
            n_prev: integer; the dimension of the previous layer            
            activation: function; the activation function for this node
            name: the name of this node 
            initialization: function; the initialization strategy to use
            use_adam: should Adam gradient descent be used to update W and b?
        """
        self.activation = activation
        self.W = initialization(n, n_prev)
        self.b = np.zeros((n, 1))
        self.keep_prob = keep_prob
        self.is_trained = False
        if use_adam:
            self.update_parameters = self.__update_adam
        else:
            self.update_parameters = self.__update_gradient_descent
        ## exponentially-weighted averages for Adam gradient descent
        self.vdW = np.zeros(self.W.shape)
        self.vdb = np.zeros(self.b.shape)
        self.sdW = np.zeros(self.W.shape)
        self.sdb = np.zeros(self.b.shape)
        
        self.name = name        
        self.A = None
        self.Z = None
    
    def shape(self):
        return self.W.shape

    def n_features(self):
        return self.shape[0]
    
    def propagate_forward_from(self, layer):
        """
        Performs forward propagation through this layer. 
        If this is layer n, then the layer argument is layer n - 1.
        """
        self.A_prev = layer.A.copy()
        self.Z = np.dot(self.W, layer.A) + self.b
        self.A = self.activation(self.Z)
        if self.keep_prob < 1 and not self.is_trained:
            self.D = (np.random.randn(self.A.shape[0], self.A.shape[1]) <
                      self.keep_prob)
            self.A *= self.D
            print(self.A)
            print(self.name)
            self.A /= self.keep_prob
            
    def propagate_backward(self, l2_scaling_factor):
        """
        Performs back propagation through this layer. 
        l2_scaling_factor: the regularization parameter lambda 
        divided by the number of training examples in the input layer.
        Zero for no regularization.
        """
        m = self.A_prev.shape[1]
        if self.keep_prob < 1:
            print("divvydiv")
            self.dA *= self.D
            self.dA /= self.keep_prob
        dZ = actv.derivative(self.activation)(self.dA, self.Z)
        self.dW = (((1 / m) * np.dot(dZ, self.A_prev.T)) +
                   ## add gradient of L2-regularized term
                   l2_scaling_factor * self.W)
        self.db = (1 / m) * np.sum(dZ, axis = 1, keepdims = True)
        return np.dot(self.W.T, dZ) ## this is dA_prev
    
    def __update_gradient_descent(self, learning_rate):
        """ 
        update parameters W and b using vanilla gradient descent.
        """
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

    def frobenius_norm(self):
        return np.sum(np.square(self.W))
        
    @staticmethod
    def correct(vec, t, beta):
        return vec / (1 - beta**t)
        
    def __update_adam(self, learning_rate, t,
               beta1, beta2, eps = 1e-8):
        """ 
        update parameters W and b using Adam gradient descent.
        """
        assert(0 <= learning_rate <= 1
               and 0 <= beta1 <= 1
               and 0 <= beta2 <= 1
               and t > 0)
        self.vdW = (beta1 * self.vdW + (1 - beta1) * self.dW)
        self.vdb = (beta1 * self.vdb + (1 - beta1) * self.db)
        self.sdW = (beta2 * self.sdW + (1 - beta2) * (self.dW**2))
        self.sdb = (beta2 * self.sdb + (1 - beta2) * (self.db**2))
        ## Apply correction to values used for dW and db update.
        vdWc = self.correct(self.vdW, t, beta1)
        sdWc = self.correct(self.sdW, t, beta2)
        vdbc = self.correct(self.vdb, t, beta1)
        sdbc = self.correct(self.sdb, t, beta2)

        self.W -= learning_rate * (vdWc / (np.sqrt(sdWc) + eps))
        self.b -= learning_rate * (vdbc / (np.sqrt(sdbc) + eps))

class Net:
    """ A Net is made of layers
    """
    def __init__(self, layer_dims, activations,
                 loss = neural_network.loss_functions.LogLoss(),
                 use_adam = False,
                 keep_prob = 1):
        """
        layer_dims: an array of layer dimensions. 
                    including the input layer.
        activations: an array of activation 
                     functions (each from the activations module); 
                     one function per layer
        loss: the cost function. 
        use_adam: should Adam gradient descent be used when training?
        """
        assert(len(layer_dims) == len(activations))

        self.use_adam = use_adam
        self.is_trained = False
        self.J = loss.cost
        self.J_prime = loss.cost_gradient
        
        self.hidden_layers = []
        for i in range(1, len(layer_dims)):
            self.hidden_layers.append(
                Layer(name = i,
                      n = layer_dims[i], n_prev = layer_dims[i - 1],
                      activation = activations[i],
                      use_adam = use_adam,
                      keep_prob = keep_prob))
            
    def __output_activation(self): return self.hidden_layers[-1].activation
            
    def __model_forward(self, input_layer):
        """ 
        Does one full forward pass through the network.        
        """    
        self.hidden_layers[0].propagate_forward_from(input_layer)
        for i in range(1, self.n_layers()):
            (self.hidden_layers[i].
             propagate_forward_from(self.hidden_layers[i - 1]))
            
    def __model_backward(self, y, l2_scaling_factor):
        """ Does one full backward pass through the network. """
        AL = self.hidden_layers[-1].A
        # derivative of cost with respect to final activation function
        dA_prev = self.J_prime(AL, y)
        for layer in reversed(self.hidden_layers):
            layer.dA = dA_prev
            dA_prev = layer.propagate_backward(l2_scaling_factor)
            
    def __update_parameters(self, learning_rate):
        """ Updates parameters on each layer at epoch t. """
        for layer in self.hidden_layers:
            layer.update_parameters(learning_rate)
            
    def __adam(self, learning_rate, t, beta1, beta2):
        """ Updates parameters on each layer at epoch t. """
        for layer in self.hidden_layers:
            layer.update_parameters(learning_rate, t, beta1, beta2)
    
    def __l2_cost(self, lambd, m):
        """ lambd: scaling parameter
            m: number of training examples
            returns: L2 cost
                     (scaled sum of frobenius norms of all hidden layer matrices)
        """
        if lambd > 0.:
            unscaled_l2_cost = np.sum([layer.frobenius_norm()
                                       for layer in self.hidden_layers])
        else:
            unscaled_l2_cost = 0
        return (lambd * unscaled_l2_cost) / (2 * m)
    
    @staticmethod
    def __get_minibatches(X, y, minibatch_size):
        """
        X: n by m matrix of training examples
        y: m length array of outcomes
        minibatch_size: size of minibatches to split into
        returns: a list of tuples [(X1_mini, y1_mini)], 
        where length of yk_mini is minibatch_size.
        """
        m = X.shape[1]
        permutation = list(np.random.permutation(m))
        X_shuffled = X[:, permutation]
        y_shuffled = y[permutation]##.reshape((1,m))
        minibatches = []
        for k in range(m // minibatch_size):
            minibatches.append(
                (X_shuffled[:, minibatch_size * k : minibatch_size * (k + 1)],
                 y_shuffled[   minibatch_size * k : minibatch_size * (k + 1)]))
        if m % minibatch_size != 0:
            minibatches.append(
                (X_shuffled[:, (m - m % minibatch_size):],
                 y_shuffled[   (m - m % minibatch_size):]))
        return minibatches

    @staticmethod
    def __check_for_edge_predictions(yhat):
        if np.max(yhat) == 1:
            warnings.warn("some predictions are 1")
        if np.min(yhat) == 0:
            warnings.warn("some predictions are 0")

    @staticmethod
    def __assert_input_ok(X):
        assert(not np.isnan(X).any())

    def __gradient_check(self, eps = 1e-7):
        """ not finished """
        W_vec  = self.__stack_things(lambda lyr: self.__matrix_to_vector(lyr.W))
        dW_vec = self.__stack_things(lambda lyr: self.__matrix_to_vector(lyr.dW))
        b_vec  = self.__stack_things(lambda lyr: lyr.b.reshape(lyr.b.shape[0]))
        db_vec = self.__stack_things(lambda lyr: lyr.db.reshape(lyr.db.shape[0]))

    def __approximate_derivative(self, vec, i, eps):
        """ not finished """
        vec[i] += eps

    @staticmethod
    def __matrix_to_vector(mat):
        """ reshape m-by-n matrix into an m*n-length array"""
        vec_len = mat.shape[0] * mat.shape[1]
        return mat.reshape(vec_len,)
    
    def __stack_things(self, action_fn):
        """ apply action_fn to each layer in hidden_layers 
            and concatenate the results into a single vector
        """
        return np.concatenate([action_fn(l) for l in self.hidden_layers])

        
    def fit(self, X, y,
            iterations = 100,
            learning_rate = 0.01,
            lambd = 0.,
            minibatch_size = 64,
            alpha_decayer = lambda epoch: 1,
            converge_at = 0.0,
            beta1 = 0.9, beta2 = 0.99,
            evaluator = None,
            debug = False):
        """ 
        Train the network.
        -- Arguments:
        If there are n features and m training examples, then:
        X: an n-by-m matrix 
        y: an array of length m
        Other arguments:
        iterations: number of times to pass through the training set.
        learning_rate: scaling factor for gradient descent step size
        alpha_decayer: function of epoch (i.e. number of passes through 
            the training set) that returns a scalar; that scalar is then
            multiplied by the learning_rate to get the decayed learning_rate.
        lambd: parameter scaling for L2 regularization
        converge_at: value of the cost function at which to stop training
        beta1, beta2: parameters that control how far back Adam-gradient-descent 
            uses on each iteration to compute the average of the gradient.
            Ignored if use_adam is False.
        debug: Should various sorts of progress information be printed?

        returns: an array of what the cost function's value was at each iteration
        """
        costs = []
        eval_values = []
        self.__assert_input_ok(X)
        input_layer = InputLayer(X)
        for i in range(1, iterations + 1):
            adj_learning_rate = alpha_decayer(i) * learning_rate
            minibatches = self.__get_minibatches(X, y, minibatch_size)
            for minibatch in minibatches:                
                (mini_X, mini_y) = minibatch
                mini_input_layer = InputLayer(mini_X)
                l2_scaling_factor = lambd / mini_input_layer.m_examples
                self.__model_forward(mini_input_layer)
                yhat = self.hidden_layers[-1].A
                self.__check_for_edge_predictions(yhat)
                cost = (self.J(yhat, mini_y) +
                        self.__l2_cost(lambd, mini_input_layer.m_examples))
                costs.append(cost)
                self.__model_backward(mini_y, l2_scaling_factor)
                if self.use_adam:
                    self.__adam(adj_learning_rate, t = i,
                                beta1 = beta1, beta2 = beta2)
                else:
                    self.__update_parameters(adj_learning_rate)
                if cost < converge_at:
                    break
            if evaluator is not None:
                eval_value = evaluator.evaluate(self)
                eval_values.append(eval_value)                
            if debug and evaluator is not None:
                train_value = evaluator.eval_fn(self, X, y)
                eval_name = evaluator.eval_metric_name
                print(f"[epoch: {i}, train {eval_name}:",
                      f"{train_value}, eval {eval_name}: {eval_value}]")
            elif debug:
                print(f"[epoch: {i}, cost: {cost}]")
        self.is_trained = True
        for layer in self.hidden_layers:
            layer.is_trained = True
        return costs, eval_values

    def predict_proba(self, X):
        """
        Use the trained network to output predictions for the examples in X.
        Precondition: is_trained must be True.
        X: an n-by-m matrix
        returns: an m-length array of predictions
        """
        ## assert(self.is_trained)
        self.__model_forward(InputLayer(X))
        yhat = self.hidden_layers[-1].A
        return np.squeeze(yhat)
    
    def n_layers(self): 
        return len(self.hidden_layers)

    def shape(self):
        """ 
        returns a list containing the shape of the weight matrix 
        in each layer 
        """
        return [l.W.shape for l in self.hidden_layers]
