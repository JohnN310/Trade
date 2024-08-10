import math 
# import pandas_datareader as web 
import yfinance as yf
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler 
from keras.models import Sequential 
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt 
plt.style.use('fivethirtyeight')

#Get the stock quote 
# df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-12-31')
df = yf.download('AAPL', start='2012-01-01', end='2019-12-31')
#Show the data 
print(df)

#Visualize the closing price 
# plt.figure(figsize=(16,8))
# plt.title('Close Price History')
# plt.plot(df['Close'])
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Close Price USD ($)', fontsize=18)
# plt.show()

#Create a new dataframe with only the "close column"
data = df.filter(['Close'])
#Convert the dataframe to a numpy array 
dataset = data.values
#Compute the number of rows to train model 1
training_data_len = math.ceil(len(dataset) * .8)
print(training_data_len)

#Scale the data 
scaler = MinMaxScaler(feature_range=(0,1)) 
scaled_data = scaler.fit_transform(dataset) 
print(scaled_data)

#Create the training dataset 
#Create the scaled trainnig dataset 
train_data = scaled_data[0:training_data_len, :]
#Split the data into x_train and y_train datasets - we want to use the first 60 data for training and we want the model to predict the 61st data valiue 
x_train = []
y_train = []
for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i <= 60: 
        print(x_train)
        print(y_train)
        print()

#Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data - because the LSTM model expects 3 dimensional data, our x_train is currently 2 dimensional
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Build the LSTM model: add is to add LSTM layers to our model's architecture
model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1],1)))
model.add(LSTM(50, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model: the optimizer is to improve upon the loss function and the loss function is to measure how well the model does.
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
