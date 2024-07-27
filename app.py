import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
from datetime import date 
import matplotlib.pyplot as plt


model = load_model('/Users/aryam/Documents/Stock Predictions Model.keras')

st.header('Stock Market Predictor')

stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2014-01-01'
end = date.today()

data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

train_split = 0.80
data_train = pd.DataFrame(data.Close[0: int(len(data)*train_split)])
data_test = pd.DataFrame(data.Close[int(len(data) * train_split): len(data)])

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0,1))


past_100_days = data_train[-100:]

# Convert past_100_days to DataFrame if it's not already
if not isinstance(past_100_days, pd.DataFrame):
    past_100_days = pd.DataFrame(past_100_days)

# Create a copy of data_test and convert it to DataFrame
data_test_copy = pd.DataFrame(data_test)

# Concatenate past_100_days and data_test_copy
data_test= pd.concat([past_100_days, data_test_copy], ignore_index=True)

data_train = scaler.fit_transform(data_train)
data_test = scaler.fit_transform(data_test)

st.subheader('Price vs 50 Day MA vs 100 Day MA vs 200 Day MA')
ma_50_days = data.Close.rolling(50).mean()
ma_100_days = data.Close.rolling(100).mean()
ma_200_days = data.Close.rolling(200).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.plot(ma_100_days, 'b')
plt.plot(ma_200_days, 'y')
plt.show()
st.pyplot(fig1)



x = []
y = []

for i in range(100, data_train.shape[0]):
    x.append(data_train[i-100:i])
    y.append(data_train[i,0])

x = np.array(x)
y = np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y * scale 

st.subheader('Price vs Predicted')

fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label = 'Actual Price')
plt.plot(y, 'g', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)