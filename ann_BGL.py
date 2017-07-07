# -*- coding: utf-8 -*-
"""
file: ann_BGL
author: Himanshu

Attributes
   1. Number of times pregnant
   2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
   3. Diastolic blood pressure (mm Hg)
   4. Triceps skin fold thickness (mm)
   5. 2-Hour serum insulin (mu U/ml)
   6. Body mass index (weight in kg/(height in m)^2)
   7. Diabetes pedigree function
   8. Age (years)
   9. Diabetes (0=no or 1=yes)
   
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### MODEL 1: PREDICTING DIABETIES YES OR NO (BINARY) ##########################

### DATA PREPROCESSING ########################################################

# import dataset
data = pd.read_csv("diabetes.csv")
diabetes = np.array(data) # convert to numpy array for slicing

# X.dtypes

X = diabetes[:, 0:8]
y = diabetes[:, 8]


# split data into training and testing datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.25, 
                                                    random_state=0)

# feature scaling (required for ANN)
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

### BUILD THE ARTIFICIAL NEURAL NETWORK #######################################

import keras
from keras.models import Sequential
from keras.layers import Dense

# initialize the ANN
classifier = Sequential()

# add first layer (inputs)
classifier.add(Dense(units = 8, 
                     kernel_initializer = 'uniform', 
                     activation = 'relu', 
                     input_dim = 8)) # 9 nodes in the hidden layer, 19 inputs 
                     #NB: (input nodes + output nodes) / 2 = units (# of nodes)

# add second layer
classifier.add(Dense(units = 8, 
                     kernel_initializer = 'uniform',
                     activation = 'relu'))

# add output layer (compare sigmoid and softmax functions)
classifier.add(Dense(units=1,
                     kernel_initializer = 'uniform',
                     activation = 'sigmoid'))

# compiling the ANN
classifier.compile(optimizer = 'adam', 
                   loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])

# fit the ANN on the training & testing set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model ########################

# Predicting the Test set results
y_pred = classifier.predict(X_test) 
# gives prediction for each observation in test set
# now in y_pred dataframe, it gives answer as true/false, rather than just probability
 
# set threshold for diabetes where p(diabeties)>0.5 means you have diabetes (1)
y_pred_bin = []
for x in y_pred:
    if x > 0.5:
        y_pred_bin.append(1)
    else: y_pred_bin.append(0)
    
# confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_bin)
cm

# accuracy
(113+38)/(113+38+24+17)
# 78.65%

'''checking chance of diabetes for a new row: e.g. patient with:
pregnant = 2
plasma glucose = 100 
Diastolic BP = 85mm Hg
Triceps skin fold = 30mm   1. Number of times pregnant
2 hr insulin = 0 mu U/ml
BMI = 35
Diabetes pedigree function = 0.8
Age = 45 years
'''
sample_patient = sc.transform(np.array([[2,100,85,30,0,35,0.8,45]]))
sample_pred = classifier.predict(sample_patient)

### MODEL 2: PREDICTING PLASMA GLUCOSE LEVEL ##################################

### DATA PREPROCESSING ########################################################

'''
Dataset description:
   1. Number of times pregnant
   2. Diastolic blood pressure (mm Hg)
   3. Triceps skin fold thickness (mm)
   4. 2-Hour serum insulin (mu U/ml)
   5. Body mass index (weight in kg/(height in m)^2)
   6. Diabetes pedigree function
   7. Age (years)
   8. Diabetes (0=no or 1=yes)
   9. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
'''

# import dataset
data2 = pd.read_csv("diabetes2.csv")
diabetes2 = np.array(data2) # convert to numpy array for slicing

# X.dtypes

X2 = diabetes2[:, 0:8]
y2 = diabetes2[:, 8]


# split data into training and testing datasets
from sklearn.model_selection import train_test_split
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, 
                                                    test_size = 0.25, 
                                                    random_state=0)

# feature scaling (required for ANN)
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X2_train = sc.fit_transform(X2_train)
X2_test = sc.fit_transform(X2_test)

### BUILD THE ARTIFICIAL NEURAL NETWORK #######################################

import keras
from keras.models import Sequential
from keras.layers import Dense

# initialize the ANN
classifier2 = Sequential()

# add first layer (inputs) - relu is best for unbounded
classifier2.add(Dense(units = 8, 
                     kernel_initializer = 'uniform', 
                     activation = 'relu', 
                     input_dim = 8)) # 9 nodes in the hidden layer, 19 inputs 
                     #NB: (input nodes + output nodes) / 2 = units (# of nodes)

# add second layer
classifier2.add(Dense(units = 8, 
                     kernel_initializer = 'uniform',
                     activation = 'relu'))

# add third layer
classifier2.add(Dense(units = 8, 
                     kernel_initializer = 'uniform',
                     activation = 'relu'))

# add output layer
classifier2.add(Dense(units=1,
                     kernel_initializer = 'uniform',
                     activation = 'linear'))

# compiling the ANN
classifier2.compile(optimizer = 'adam', 
                   loss = 'mean_squared_logarithmic_error', 
                   metrics = ['accuracy'])

# fit the ANN on the training & testing set
classifier2.fit(X2_train, y2_train, batch_size = 10, epochs = 100)

'''   
All activation functions: https://keras.io/activations/
All loss functions: https://keras.io/losses/
'''

# Part 3 - Making predictions and evaluating the model ########################

# Predicting the Test set results
y2_pred = classifier2.predict(X2_test) 

# clark error grid

# build the plot grid
# NB: predicted values must be on the Y axis
from pylab import *
import matplotlib.pyplot as plt
plt.xlim(0,250)
plt.ylim(0,250)
plt.title('Clark Error Grid of Algorithm Accuracy')
plt.xlabel('Actual Plasma Glucose Level')
plt.ylabel('Predicted Plasma Glucose Level')

fill([38.5,250,250,38.5], [0,0,198,32], 'b', alpha=0.2, edgecolor='r') #B
fill([0,32,210,0], [38.5,38.5,250,250], 'b', alpha=0.2, edgecolor='r') #B

fill([135,250,250,135], [38.5,38.5,100,100], 'c', alpha=0.5, edgecolor='g') #D
fill([0,32,32,0], [38.5,38.5,100,100], 'c', alpha=0.5, edgecolor='g') #D

fill([100,250,250,100], [0,0,38.5,38.5], 'b', alpha=0.4, edgecolor='r') #E
fill([0,32,32,0], [100,100,250,250], 'b', alpha=0.4, edgecolor='r') #E

fill([80,100,100], [0,0,38.5], 'b', alpha=0.2, edgecolor='r') #C
fill([32,187.5,32], [100,250,250], 'b', alpha=0.2, edgecolor='r') #C

# insert scatter points
clark = plt.scatter(y2_test, y2_pred)

### MODEL 3: PREDICTING PLASMA GLUCOSE WITHOUT DIABETES VARIABLe###############

### DATA PREPROCESSING ########################################################

'''
Dataset description:
   1. Number of times pregnant
   2. Diastolic blood pressure (mm Hg)
   3. Triceps skin fold thickness (mm)
   4. 2-Hour serum insulin (mu U/ml)
   5. Body mass index (weight in kg/(height in m)^2)
   6. Age (years)
   7. Diabetes (0=no or 1=yes)
   8. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
   
Diabetes pedigree function is taken out 
'''

# import dataset
data3 = pd.read_csv("diabetes3.csv")
diabetes3 = np.array(data3) # convert to numpy array for slicing

# split X and y
X3 = diabetes3[:, 0:7]
y3 = diabetes3[:, 7]


# split data into training and testing datasets
from sklearn.model_selection import train_test_split
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, 
                                                    test_size = 0.25, 
                                                    random_state=0)

# feature scaling (required for ANN)
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X3_train = sc.fit_transform(X3_train)
X3_test = sc.fit_transform(X3_test)

### BUILD THE ARTIFICIAL NEURAL NETWORK #######################################

import keras
from keras.models import Sequential
from keras.layers import Dense

# initialize the ANN
classifier3 = Sequential()

# add first layer (inputs) - relu is best for unbounded
classifier3.add(Dense(units = 7, 
                     kernel_initializer = 'uniform', 
                     activation = 'relu', 
                     input_dim = 7)) # 9 nodes in the hidden layer, 19 inputs 
                     #NB: (input nodes + output nodes) / 2 = units (# of nodes)

# add second layer
classifier3.add(Dense(units = 7, 
                     kernel_initializer = 'uniform',
                     activation = 'relu'))

# add third layer
classifier3.add(Dense(units = 7, 
                     kernel_initializer = 'uniform',
                     activation = 'relu'))

# add output layer
classifier3.add(Dense(units=1,
                     kernel_initializer = 'uniform',
                     activation = 'linear'))

# compiling the ANN
classifier3.compile(optimizer = 'adam', 
                   loss = 'mean_squared_error', 
                   metrics = ['accuracy'])

# fit the ANN on the training & testing set
classifier3.fit(X3_train, y3_train, batch_size = 10, epochs = 100)

'''   
All activation functions: https://keras.io/activations/
All loss functions: https://keras.io/losses/
'''

# Part 3 - Making predictions and evaluating the model ########################

# Predicting the Test set results
y3_pred = classifier3.predict(X3_test) 

# clark error grid

# build the plot grid
# NB: predicted values must be on the Y axis
from pylab import *
import matplotlib.pyplot as plt
plt.xlim(0,250)
plt.ylim(0,250)
plt.title('Clark Error Grid of Algorithm Accuracy')
plt.xlabel('Actual Plasma Glucose Level')
plt.ylabel('Predicted Plasma Glucose Level')

fill([38.5,250,250,38.5], [0,0,198,32], 'b', alpha=0.2, edgecolor='r') #B
fill([0,32,210,0], [38.5,38.5,250,250], 'b', alpha=0.2, edgecolor='r') #B

fill([135,250,250,135], [38.5,38.5,100,100], 'c', alpha=0.5, edgecolor='g') #D
fill([0,32,32,0], [38.5,38.5,100,100], 'c', alpha=0.5, edgecolor='g') #D

fill([100,250,250,100], [0,0,38.5,38.5], 'b', alpha=0.4, edgecolor='r') #E
fill([0,32,32,0], [100,100,250,250], 'b', alpha=0.4, edgecolor='r') #E

fill([80,100,100], [0,0,38.5], 'b', alpha=0.2, edgecolor='r') #C
fill([32,187.5,32], [100,250,250], 'b', alpha=0.2, edgecolor='r') #C

# insert scatter points
clark = plt.scatter(y3_test, y3_pred)