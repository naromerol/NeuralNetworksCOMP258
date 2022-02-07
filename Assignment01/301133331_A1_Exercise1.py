"""
COMP258 - NEURAL NETWORKS
Assignment 1 - Exercise 1
Nestor Romero - 301133331
"""

from turtle import shape
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


class Basic_Perceptron:
    """
    Perceptron base class to implement AND logic gate
    """
    #DEBUG LEVELS
    NONE = 0
    TRAINING = 1
    PREDICT = 2
    def __init__(self, num_inputs, debug_level=0, bias=0, learning_rate=1):
        # initialize input layer and weights with zeros
        self.num_inputs = num_inputs
        self.input_values = np.zeros(num_inputs)
        self.weights = np.zeros(num_inputs)

        # configure additional parameters
        # 0 None, 1 Training, 2 Predict
        self.debug_level = debug_level
        self.bias = 0
        self.learning_rate = learning_rate

    def train(self, input_values, expected, activation_type='step'):
        self.input_values = input_values
        predicted = self.predict(self.input_values, activation_type)

        if self.debug_level >= Basic_Perceptron.TRAINING:
            print(f'predicted: {predicted}, expected: {expected}')
        
        if(predicted != expected):
            #Calculate difference in results
            diff = expected - predicted
            
            for i in range(self.num_inputs):
                self.weights[i] += diff * input_values[i]
                self.bias += diff

    def predict(self, input_values, activation_type='step'):
        # calculate the weighted sum
        
        weighted_sum = np.dot(input_values, self.weights) + self.bias
        result = self.calculate_activation(weighted_sum, activation_type)
        if self.debug_level >= Basic_Perceptron.PREDICT:
            print(f'input: {input_values}, weights: {self.weights}, weighted_sum: {weighted_sum}, result: {result}')
        return result

    def calculate_activation(self, wsum, type='Step'):
        if(type.lower() == 'step'):
            if(wsum >= 1):
                return 1
            return 0
        if(type.lower() == 'sigmoid'):
            return 1.0/(1.0+np.exp(-wsum))
        return 0
    
    def simulate_train_data(self, num_iters):
        num_iterations = 10
        for _ in range(num_iterations):
            x1 = np.random.randint(2,size=1)[0]
            x2 = np.random.randint(2,size=1)[0]
            y = x1 and x2
            print(f'training instance ({x1},{x2} => {y})')
            self.train(np.array([x1,x2]),y)    

# Program execution
######
######
######
######      Part 1a. AND Logic gate
######
######
 
print('\n\nPART 1A. ANG LOGIC GATE')
and_p = Basic_Perceptron(2, debug_level=Basic_Perceptron.TRAINING)

#Train the perceptron
num_iters = 10
print('TRAINING PHASE')
and_p.simulate_train_data(num_iters)

#Test the perceptron
print('\n\nTESTING PHASE')
y_actual = [1,0,0,0]
y_pred = []
y_pred.append(and_p.predict(np.array([1,1])))
y_pred.append(and_p.predict(np.array([1,0])))
y_pred.append(and_p.predict(np.array([0,1])))
y_pred.append(and_p.predict(np.array([0,0])))
print(y_actual)
print(y_pred)
accuracy = accuracy_score(y_actual, y_pred)
print(f'Accuracy: {accuracy} for {num_iters} training iterations')

"""
RESULTS EXERCISE 1A: 
Generally an accuracy of 75% was achieved but it is quite bad given the exercise and a high chance of a zero (False) value
The regular perceptron lerning rule makes it difficult to estabilize the right weights
"""

######
######
######
######      Part 2b. Analyze Iris Data
######
######

print('\n\nPART 1B. PERCEPTRON WITH IRIS DATA')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv('iris.data', header=0, names=['sepl', 'sepw', 'petl', 'petw', 'class'])
# print(data.head())
# print(data.info())
# print(data.describe())  # 149 data points
# print(data['class'].value_counts())
# Iris-versicolor    50 : Class 0
# Iris-virginica     50 : Class 1
# Iris-setosa        49 >> selected to be removed from exercise set and manage only two expecteds

data_prepared = data[data['class'] != 'Iris-setosa']
# print(data_prepared.describe())
#print(data_prepared['class'].value_counts())

print(data_prepared['class'].unique())
lbl_encoder = LabelEncoder()
lbl_encoder.fit(data_prepared['class'].unique())

#Create training and test sets
X = data_prepared[data_prepared.columns[:-1]]
y_raw = data_prepared[data_prepared.columns[-1]]

#Codify classes in 0 and 1
y = lbl_encoder.transform(y_raw)
#PSlit data in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=301133331)

iris_p = Basic_Perceptron(4, debug_level=Basic_Perceptron.NONE)
y_pred = np.zeros(len(y_test))

print('\n\nTRAINING PHASE')
for index in range(len(X_train)): 
    iris_p.train(X_train.iloc[index].values, y_train[index])

print('\n\nTESTING PHASE')
for index in range(len(X_test)): 
    y_pred[index] = iris_p.predict(X_test.iloc[index].values)

accuracy = accuracy_score(y_test,y_pred)
print(f'Accuracy for iris data: {accuracy}')