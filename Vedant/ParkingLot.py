# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 18:30:14 2018

@author: Vedant
"""



#importing the libraries
import numpy as np
import  matplotlib.pyplot as plt
import pandas as pd


#importing the dataset:

dataset_train = pd.read_csv('hourly_data_train.csv')

#1:2 is used to use it as a numpy array and not as a series

training_set = np.array(dataset_train.iloc[:,1:2])


#Feature scaling:
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
scaled_training_set = sc.fit_transform(training_set)

#making a data structure to store the data for 60 previous time stamps and 1 output
#x_train will store data for previous 60 observations and y_train will store data for 1 observation:

x_train = []
y_train = []

for i in range(60,3598):
    x_train.append(scaled_training_set[i-60:i,0])
    y_train.append(scaled_training_set[i,0])

x_train,y_train = np.array(x_train), np.array(y_train)

#Reshaping our input:

print(x_train.shape[0])
print(x_train.shape[1])
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1], 1))


#Building Rnn

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

#adding the first LSTM layer
regressor.add(LSTM(units = 50,return_sequences= True, input_shape = (x_train.shape[1],1), dropout=0.2))

#adding the second layer
regressor.add(LSTM(units = 50,return_sequences= True, dropout=0.2))


#adding the third layer
regressor.add(LSTM(units = 50,return_sequences= True, dropout=0.2))


#adding the fourth layer
regressor.add(LSTM(units = 50, dropout=0.2))

#adding the last layer
regressor.add(Dense(units = 1 ))

#Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#fitting the RNN to the training data set:
regressor.fit(x_train,y_train,epochs = 200, batch_size = 25 )

#saving the model for future use:
#this prevents the retraining of the model
from keras.models import load_model
regressor.save('my_model.h5') 

#making predictions and visualizing the results:

#getting the stock price from test set
dataset_test = pd.read_csv('hourly_data_test.csv')
test_set = np.array(dataset_test.iloc[:,1:2])

#getting the predicted stock price:

#combining both the datasets as in the input we need to give the data 
#of past 60 days as the input
dataset_total = pd.concat((dataset_train['available_spots'], dataset_test['available_spots']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

#building the test data frame:
x_test = []


for i in range(60,114):
    x_test.append(inputs[i-60:i,0])
    
x_test= np.array(x_test)

#Reshaping our input:

x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1], 1))  
predicted_stock_price = regressor.predict(x_test)


#calculating the accuracy of prediction:
#calculating the mean sqaured error:

error = 0


for i in range(len(predicted_stock_price)):
    
    error = error + np.sqrt(np.square(predicted_stock_price[i] - dataset_test.iloc[i,1]))

print("Error:",error)
print("Mean Squared Error:", error/len(predicted_stock_price))



    
#removing the scalling to get the actual stock prices
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


#Visualizing the results
plt.plot(test_set , color = 'red', label = 'Real Parking Space')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Parking Space')
plt.title('Parking lot prediction')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Number of Time Slots')
plt.show()




    