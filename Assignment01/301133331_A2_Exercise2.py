"""
COMP258 - NEURAL NETWORKS
Assignment 1 - Exercise 2
Nestor Romero - 301133331
"""

import random
import numpy as np

class Hepatitis_ANN(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))
    
    def sigmoid_prime(self,z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a)+b)
        return a

    def Execute(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        # Size of training data
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            
            random.shuffle(training_data)
            
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test));
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        '''Error: data structure needs adjustment'''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
    def backprop(self, x, y):
        '''Error: data structure needs adjustment'''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def evaluate(self, test_data):
        '''Calculate result '''
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
    
    
######
######
######
######      Exercise 2. Predict Die_Live
######
######

#Load data
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

train_data = pd.read_json('hepatitis_training_data.json')
test_data = pd.read_json('hepatitis_testing_data.json')

# print(train_data.info())
# print(train_data.describe())
# print(len(train_data)) # 152 data points, no missing values detected, all columns counts = 152
# print(len(test_data)) # 3 data points

#Recodify output in data (0,1)
train_data[train_data['Die_Live']==1] = 0
train_data[train_data['Die_Live']==2] = 1
test_data[test_data['Die_Live']==1] = 0
test_data[test_data['Die_Live']==2] = 1

#Create train and test X,y pairs
X_train = train_data[train_data.columns[:-1]]
y_train = train_data[train_data.columns[-1]]
X_test = train_data[test_data.columns[:-1]]
y_test = train_data[test_data.columns[-1]]

#Hepatitis ANN
haan = Hepatitis_ANN([19, 30, 15, 2])
haan.Execute(zip(X_train, y_train), 1, 50, 3.0, test_data=zip(X_test, y_test))