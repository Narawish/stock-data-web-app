# stock-data-web-app

in order to run the webpage use 'streamlit run app.py' in this directory

## Problem found and Fixing Method
1. pandas_datareader
In the video, the instructer used pdr.DataReader but this method cannot be used with unknown reasons.
The fixing method of this via surfing the stackoverflow:

from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()
df = pdr.get_data_yahoo('TSLA', start ,end)

2. Dataframe has not append method.
In the video, the instructor used append to combine the past 100 days data and testing data  but it cannot be used here.
Fixing method is using pd.concat()

final_df = pd.concat([past_100_days,data_testing], axis=0)

3. Streamlit run app.py getting ModuleNotFoundError: No module named 'streamlit.cli'
Fixing method => pip install --upgrade streamlit


## Lesson Learned
1. Basic of LSTM
2. Basic of Streamlit library

## Self learning
1. Adding dynamic date range using streamlit date picker (st.date_input())
2. Adding dynamics chart showing depending on the number of total days
3. Adding tabs into the webpage for easier view