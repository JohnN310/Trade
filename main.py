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
df = yf.download('AAPL', start='2012-01-01', end='2019-12-19')
#Show the data 
print(df)

#Visualize the closing price 
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()

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

# Compile the model: the optimizer is to improve upon the loss function and the loss function is to measure how well the model does on training.
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Train the model: epochs is the number of iterations for the entire dataset is passed forward and backwards 
model.fit(x_train, y_train, batch_size = 1, epochs = 1)

#Create the testing dataset 
#Create a new array containing scaled values from index 1543 to 2003: this is for the scaled testing dataset 
test_data = scaled_data[training_data_len - 60:, :]
#Create the data sets x_test and y _test 
x_test = []
y_test = dataset[training_data_len:, :] #unscaled, actual values in the dataset
for i in range (60, len(test_data)):
    x_test.append(test_data[i-60:i,0])

#Convert the data to a numpy array 
x_test = np.array(x_test)

#Reshape the data: x_test.shape[0] is the number of samples, x_test.shape[1] is the number of time stamps, and the number of features, which is the close price, is 1
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Get the model's predeicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions) # since predictions currently hold the scaled predicted price, we have to inverse transform it to the actual price.
# here we want predictions to contain the same values as y_test values

#Get the root mean squared error (RMSE) - the lower value indicates a better fit, a value of 5 is pretty decent. 
rmse = np.sqrt(np.mean(predictions - y_test)**2)
print(rmse)

#Plot the data 
train = data[0:training_data_len]
validation = data[training_data_len:]
validation['Predictions'] = predictions
#Visualize the data 
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date',fontsize = 18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(validation[['Close', 'Predictions']])
plt.legend(['Train', 'Validation', 'Predictions'], loc = 'lower right')
plt.show()


#Show the valid and predicted prices 
print(validation)
###################################################################################################################################
#Below is to get a single predicted price
#Get the quote 
apple_quote = yf.download('AAPL', start='2012-01-01', end='2019-12-17')

#Create a new dataframe 
new_df = apple_quote.filter(['Close'])

#Get the last 60 days values and convert the dataframe to an np array 
last_60_days = new_df[-60:].values

#Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days) # we are not using the fit transform because we want to transform the data using the min and max values 

#Creatre an empty test list and append the last 60 days data to the test list 
X_test = []
X_test.append(last_60_days_scaled)

#Convert the dataset to an np array 
X_test = np.array(X_test)

#Reshape the data 
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#Get the predicted scaled price for 12/18 
pred_price = model.predict(X_test)

#Undo the scaling 
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)

#Get the actual price 
apple_quote_2 = yf.download('AAPL', start='2019-12-18', end='2019-12-19')
print(apple_quote_2['Close'])