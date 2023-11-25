import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
import datetime as dt
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

yf.pdr_override()



#Setting the title of webpage
st.title('Stock Trend Prediction')

#Receiving user input (setting the default value as 'AAPL)
user_input = st.text_input('Enter Stock Ticker','AAPL')
#Receiving userdate input to make more dynamic web
start = st.date_input("Starting date")
end = st.date_input('Ending date')

if end <= dt.date.today():
    total_days = (end - start).days
    st.write('total days = '  , total_days)
    df = pdr.get_data_yahoo(user_input, start ,end)

    #Describing data
    st.subheader(str(user_input)+ ' Data from '+ str(start)+ ' - '+ str(end))
    st.write(df.describe())

    #Visualization
    st.subheader('Closing Price vs Time Chart of '+str(user_input))
    fig = plt.figure(figsize = (12,6))
    plt.plot(df.Close)
    st.pyplot(fig)

    if total_days >= 100:
        st.subheader('Closing Price vs Time Chart with 100 MA of '+str(user_input))
        ma100 = df.Close.rolling(100).mean()
        fig = plt.figure(figsize = (12,6))
        plt.plot(df.Close, label = 'Closing')
        plt.plot(ma100, c='r', label = 'MA100')
        plt.legend()
        st.pyplot(fig)

        if total_days >= 200:
            st.subheader('Closing Price vs Time Chart with 100 MA & 200 MA of '+str(user_input))
            ma100 = df.Close.rolling(100).mean()
            ma200 = df.Close.rolling(200).mean()
            fig = plt.figure(figsize = (12,6))
            plt.plot(df.Close, label = 'Closing')
            plt.plot(ma100, c='r', label = 'MA100')
            plt.plot(ma200, c='g', label = 'MA200')
            plt.legend()
            st.pyplot(fig)

        if total_days > 148:
            #ML Part
            # Splitting into training and testing data

            data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
            data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

            #Scale the feature

            scaler = MinMaxScaler(feature_range=(0,1))
            data_training_array = scaler.fit_transform(data_training)


            #Load model
            model = load_model('keras_model.h5')

            #Testing Part
            final_df = pd.concat([data_training.tail(100),data_testing], axis=0)
            input_data = scaler.fit_transform(final_df)

            X_test = []
            y_test = []

            for i in range(100, input_data.shape[0]):
                X_test.append(input_data[i-100:i])
                y_test.append(input_data[i,0])

            X_test, y_test = np.array(X_test), np.array(y_test)


            #Making predictions
            y_predicted = model.predict(X_test)

            scale_factor = 1/scaler.scale_[0]
            y_predicted = y_predicted * scale_factor
            y_test = y_test*scale_factor


            #LSTM Graph
            st.subheader('Closing Price Prediction vs Original using LSTM')
            fig = plt.figure(figsize = (12,6))
            plt.plot(y_test,'b',label = 'Original Price')
            plt.plot(y_predicted, 'r', label = 'Predicted Price')
            plt.xlabel('Time (days)')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(fig)

elif end > dt.date.today():
    st.write('Cannot see the future price')