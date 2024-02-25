import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as pdr
import yfinance as yf
from datetime import datetime
from keras.models import load_model
import streamlit as st
yf.pdr_override()
st.title('Trade Insight')
user_input=st.text_input("Enter Stock Name Code: ","AAPL")
startdate=st.text_input("Enter Start Date:","2010-1-1")
endate=st.text_input("Enter Start Date:","2024-2-24")
data=pdr.get_data_yahoo(user_input,start=startdate,end=endate)

st.subheader('Data from '+startdate+' to '+endate+'.')
st.write(data)

st.subheader('Descriptive Statistics:- ')
st.write(data.describe())

st.subheader('Opening Price Vs Time Chart: ')
fig=plt.figure(figsize=(12,8))
plt.plot(data.Open)
st.pyplot(fig)

st.subheader('Opening Price Vs Time Chart with 100 moving avg:')
m100=data.Open.rolling(100).mean()
fig=plt.figure(figsize=(12,8))
plt.plot(m100)
plt.plot(data.Open)
st.pyplot(fig)

st.subheader('Opening Price Vs Time Chart with 100 moving avg. and 200 moving avg. :')
m100=data.Open.rolling(100).mean()
m200=data.Open.rolling(200).mean()
fig=plt.figure(figsize=(12,8))
plt.plot(m100,"b")
plt.plot(m200,"r")
plt.plot(data.Open,"g")
st.pyplot(fig)


#splitting data into Training and Testing
data_train=pd.DataFrame(data['Open'][0:int(len(data)*0.70)])
data_test=pd.DataFrame(data['Open'][int(len(data)*0.70):int(len(data))])



from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_train_array=scaler.fit_transform(data_train)


model=load_model('model.h5')

past_100_days=data_train.tail(100)

final_data = pd.concat([past_100_days, data_test], ignore_index=True)

input_data=scaler.fit_transform(final_data)

x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)


y_predicted=model.predict(x_test)


y_predicte=scaler.inverse_transform(y_predicted)
y_test=y_test.reshape(-1,1)
my_test=scaler.inverse_transform(y_test)

st.subheader("Predictions Vs. Original:-")
figure=plt.figure(figsize=(12,6))
plt.plot(my_test,'b',label='Original Price')
plt.plot(y_predicte,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('price')
plt.legend()
st.pyplot(figure)

st.subheader("RMSE is: ")
from sklearn.metrics import mean_squared_error
rmse=np.sqrt(mean_squared_error(y_test,y_predicted))
st.write("The root mean squared error is {}.".format(rmse))