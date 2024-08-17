# --------------------------------------------- LIBRARIES --------------------------------------------- #

# MATH
import math
import pandas as pd
import numpy as np
from datetime import date, timedelta

# VISUALIZATION
import plotly.graph_objs as go
import plotly.express as px
import plotly

# DATA
import yfinance as yf
from ta.trend import MACD

# MACHINE LEARNING
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.layers import Dense, LSTM
from keras.models import Sequential

# WEB COMPONENTS
import streamlit as st
import streamlit_option_menu as som
from streamlit_extras.metric_cards import style_metric_cards

# ------------------------------------------- HIDDEN WEB ELEMENTS ------------------------------------------- #

st.set_page_config(
  page_title=' Ethereum Price Prediction App',
  page_icon=':bar_chart:',
  layout='wide'
)

hide_st_style = """
  <style>
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  </style>
  """ 

# LOAD CSS 
with open('style.css') as f:
  st.markdown(
    f"<style>{f.read()}</style>", 
    unsafe_allow_html=True
    )

# STATING PREDICT BUTTON TO UNCLICKED
if 'clicked' not in st.session_state:
  st.session_state.clicked = False

# --------------------------------------------- HELPER FUNCTION --------------------------------------------- #

# RETRIEVING DATASET
@st.cache_data()
def load_data(ticker):
  df = yf.download(tickers=ticker, period='10y', interval='1d')
  return df

# CREATE NEW DATA FORMAT
def new_dataset(dataset, step_size):
  x, y = [], []
  for i in range(len(dataset) - step_size - 1):
    a = dataset[i:(i + step_size), 0]
    x.append(a)
    y.append(dataset[i + step_size, 0])
  return np.array(x), np.array(y)

# CHANGING CLICKED STATE
def click_button():
  st.session_state.clicked = True

# PREDICTION
def start_predict(time_of_prediction, prediction_method, dataset):

  # CREATE PROGRESS BAR
  progress_bar = st.progress(0)
  progress_text = st.empty()

  # DATES
  date_start = (date.today() - timedelta(days=30))
  date_end = date.today()
  date_new = (date.today() + timedelta(days=1))
  date_last = (date.today() + timedelta(days=30))

  # DATASET
  df = dataset
  df = df.reset_index()
  df_dates = pd.to_datetime(df['Date']).dt.date

  if prediction_method == 'Short Term':
    df0 = df[['Close', 'Open', 'High', 'Low']]
  elif prediction_method == 'Long Term':
    df0 = df[['Adj Close', 'Open', 'High', 'Low']]

  # df0 = df[['Close', 'Open', 'High', 'Low']]
  # st.write(df0.columns)
  df1 = df0.mean(axis=1)

  # SCALING/NORMALIZING DATA
  scaler = MinMaxScaler((0, 1))
  df2 = np.reshape(df1.values, (len(df1), 1))
  df3 = scaler.fit_transform(df2)

  # SPLITTING TRAINING/TESTING DATA
  df_train = int(len(df3) * .8)
  df_test = int(len(df3) - df_train)
  df_train, df_test = df3[0:df_train,:], df3[df_train:len(df3),:]

  step_size = 90
  X_train, Y_train = new_dataset(df_train, step_size)
  X_test, Y_test = new_dataset(df_test, step_size)

  # MODELLING
  model = Sequential()
  model.add(LSTM(units=50, return_sequences=True, input_shape=(step_size, 1)))
  model.add(LSTM(units=50))
  model.add(Dense(1))

  # TRAINING
  model.compile(optimizer='adam', loss='mean_squared_error')
  num_epochs = 10
  # history = model.fit(X_train, Y_train, epochs=num_epochs, batch_size=32)

  # PROGRESSING THE TRAINING
  for epoch in range(num_epochs):
    
    # FITTING MODEL FOR EACH EPOCH
    history = model.fit(X_train, Y_train, epochs=1, batch_size=32, verbose=2)

    # UPDATE THE PROGRESS BAR
    progress_percentage = (epoch + 1) / num_epochs
    progress_bar.progress(progress_percentage)
    progress_text.text('Predicting Progress: %.0f' % (progress_percentage * 100) + '%')

  # PREDICTING RESULT
  pred_train = model.predict(X_train)
  pred_test = model.predict(X_test)

  score_train = math.sqrt(mean_squared_error(Y_train, pred_train))
  score_test = math.sqrt(mean_squared_error(Y_test, pred_test))

  print('RMSE (Train): %.2f' % (score_train))
  print('RMSE (Test): %.2f' % (score_test))

  # INVERSING SCALED DATA TO GET REAL VALUE
  inv_pred_train = scaler.inverse_transform(pred_train)
  inv_Y_train = scaler.inverse_transform([Y_train])
  inv_pred_test = scaler.inverse_transform(pred_test)
  inv_Y_test = scaler.inverse_transform([Y_test])

  # PREDICTING n DAYS PRICE INTO THE FUTURE
  n_future = time_of_prediction
  y_future = []

  pred_x = X_test[-1:, :]
  pred_y = pred_test[-1]

  for i in range(n_future):

    # FEED THE LAST FORECAST BACK TO THE MODEL AS AN INPUT
    pred_x = np.append(pred_x[:, 1:], pred_y.reshape(1, 1), axis=1)

    # GENERATE NEXT FORECAST
    pred_y = model.predict(pred_x)

    # SAVE FORECAST TO THE LIST
    y_future.append(pred_y.flatten()[0])

  y_future = np.array(y_future).reshape(-1, 1)
  inv_y_future = scaler.inverse_transform(y_future)

  # SHOWING THE RESULT OF THE PREDICTION
  df_avg_90 = pd.DataFrame(columns=['Date', 'Average Predicted Price'])
  df_avg_90 ['Date'] = pd.date_range(date_end + pd.Timedelta(days=1), periods=n_future)
  df_avg_90['Average Predicted Price'] = inv_y_future
  df_avg_90.reset_index()

  df_actual_avg = df0.copy()
  df_actual_avg['Average Price (Actual)'] = df_actual_avg.mean(numeric_only=True, axis=1)
  df_actual_avg_date = df['Date']
  df_actual_avg.insert(0, 'Date', df_actual_avg_date)

  if prediction_method == 'Short Term':
    df_actual_avg.drop(['Close', 'Open', 'High', 'Low'], axis=1, inplace=True)
  elif prediction_method == 'Long Term':
    df_actual_avg.drop(['Adj Close', 'Open', 'High', 'Low'], axis=1, inplace=True)

  tmp = (df_dates > date_start)
  df_actual_avg = df_actual_avg.loc[tmp]
  df_actual_avg.reset_index()

  tmp = [df_actual_avg, df_avg_90]
  df_actual_vs_avg_90 = pd.concat(tmp)
  
  # st.subheader("Average Predicted Price for The Next " + str(time_of_prediction) + " Days From Today")
  # st.write(df_actual_vs_avg_90)
  # print("Prediction Accuracy is: %.2f" % (100 - score_test) + "%")
  print('Prediction done')
  st.success('Prediction Done!')
  st.session_state.clicked = False
  return df_actual_vs_avg_90

# SHOWING CURRENT STOCK INFORMATION
def metrics(data):
  
  # date_today = date.today()
  # date_before_today = date_today - timedelta(days=1)
  d = data.reset_index()

  # Sort the DataFrame by 'Date' column in descending order
  d_sorted = d.sort_values(by='Date', ascending=False)
  
  # today's date
  today_date = d_sorted.iloc[0]['Date'].strftime('%Y-%m-%d')

  # Get the second row, which represents the date before the latest date
  previous_date = d_sorted.iloc[1]['Date'].strftime('%Y-%m-%d')

  st.subheader('Current Ethereum Market Stocks Information (' + str(today_date) + ')')

  col1, col2, col3, col4, col5, col6 = st.columns(6)
  
  # open
  col1.metric(
    label='Open', 
    value=d_sorted.loc[d_sorted['Date'] == d_sorted.iloc[0]['Date'], 'Open'], 
    delta= int(d_sorted.loc[d_sorted['Date'] == d_sorted.iloc[0]['Date'], 'Open']) - int(d_sorted.loc[d_sorted['Date'] == previous_date, 'Open'])
    )

  # close
  col2.metric(
    label='Close', 
    value=d_sorted.loc[d_sorted['Date'] == d_sorted.iloc[0]['Date'], 'Close'], 
    delta= int(d_sorted.loc[d_sorted['Date'] == d_sorted.iloc[0]['Date'], 'Close']) - int(d_sorted.loc[d_sorted['Date'] == previous_date, 'Close'])
    )

  # high
  col3.metric(
    label='High', 
    value=d_sorted.loc[d_sorted['Date'] == d_sorted.iloc[0]['Date'], 'High'], 
    delta= int(d_sorted.loc[d_sorted['Date'] == d_sorted.iloc[0]['Date'], 'High']) - int(d_sorted.loc[d_sorted['Date'] == previous_date, 'High'])
    )

  # low
  col4.metric(
    label='Low', 
    value=d_sorted.loc[d_sorted['Date'] == d_sorted.iloc[0]['Date'], 'Low'], 
    delta= int(d_sorted.loc[d_sorted['Date'] == d_sorted.iloc[0]['Date'], 'Low']) - int(d_sorted.loc[d_sorted['Date'] == previous_date, 'Low'])
    )

  # adj close
  col5.metric(
    label='Adj Close', 
    value=d_sorted.loc[d_sorted['Date'] == d_sorted.iloc[0]['Date'], 'Adj Close'], 
    delta= int(d_sorted.loc[d_sorted['Date'] == d_sorted.iloc[0]['Date'], 'Adj Close']) - int(d_sorted.loc[d_sorted['Date'] == previous_date, 'Adj Close'])
    )

  # volume
  col6.metric(
    label='Volume', 
    value=d_sorted.loc[d_sorted['Date'] == d_sorted.iloc[0]['Date'], 'Volume'], 
    delta= int(d_sorted.loc[d_sorted['Date'] == d_sorted.iloc[0]['Date'], 'Volume']) - int(d_sorted.loc[d_sorted['Date'] == previous_date, 'Volume'])
    )

  return 0

# Chart: HISTORICAL PRICE TREND
def plot_historical_price_trend(data):

  # create traces
  trace1 = go.Scatter(x=data.index, y=data['Open'], mode='lines', name='Open')
  trace2 = go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close')
  trace3 = go.Scatter(x=data.index, y=data['High'], mode='lines', name='High')
  trace4 = go.Scatter(x=data.index, y=data['Low'], mode='lines', name='Low')

  fig = go.Figure(data=[trace1, trace2, trace3, trace4])
  fig.update_layout(title='Ethereum Historical Price Trend', xaxis_title='Date', yaxis_title='Price (USD)')
  
  # add date slider
  fig.update_xaxes(
    rangeslider_visible=True
  )

  # show plot
  st.plotly_chart(fig, use_container_width=True)

# Chart: MOVING AVERAGES
def plot_moving_averages(data):
  data['MA50'] = data['Close'].rolling(window=50).mean()
  data['MA100'] = data['Close'].rolling(window=100).mean()
  data['MA200'] = data['Close'].rolling(window=200).mean()

  # Create figure
  fig = go.Figure()

  # Add traces for closing price and moving averages
  fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Closing Price'))

  fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], mode='lines', name='50-Day MA'))
  fig.add_trace(go.Scatter(x=data.index, y=data['MA100'], mode='lines', name='100-Day MA'))
  fig.add_trace(go.Scatter(x=data.index, y=data['MA200'], mode='lines', name='200-Day MA'))

  # Add shaded regions to indicate potential areas of support and resistance
  fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], fill='tonexty', fillcolor='rgba(0,100,80,0.2)', mode='none', name='50-Day MA Support/Resistance'))
  fig.add_trace(go.Scatter(x=data.index, y=data['MA100'], fill='tonexty', fillcolor='rgba(0,176,246,0.2)', mode='none', name='100-Day MA Support/Resistance'))
  fig.add_trace(go.Scatter(x=data.index, y=data['MA200'], fill='tonexty', fillcolor='rgba(231,107,243,0.2)', mode='none', name='200-Day MA Support/Resistance'))

  # Update layout
  fig.update_layout(title='Ethereum Moving Averages', xaxis_title='Date', yaxis_title='Price (USD)', showlegend=True)

  # Add annotations for each moving average
  fig.update_layout(annotations=[
      dict(x=data.index[-1], y=data['MA50'].iloc[-1], xref='x', yref='y', text='50-Day MA', showarrow=True, arrowhead=2, ax=20, ay=-40),
      dict(x=data.index[-1], y=data['MA100'].iloc[-1], xref='x', yref='y', text='100-Day MA', showarrow=True, arrowhead=2, ax=20, ay=-40),
      dict(x=data.index[-1], y=data['MA200'].iloc[-1], xref='x', yref='y', text='200-Day MA', showarrow=True, arrowhead=2, ax=20, ay=-40)
  ])

  # add date slider
  fig.update_xaxes(
    rangeslider_visible=True
  )

  # show plot
  st.plotly_chart(fig, use_container_width=True)

# Chart: VOLUME TRADED
def plot_trading_volume(data):
  # Calculate average volume
  avg_volume = data['Volume'].mean()

  # Create a new column to indicate whether the volume is above or below average
  data['Volume Status'] = ['Above Average' if vol > avg_volume else 'Below Average' for vol in data['Volume']]

  # Create a line chart for average volume
  avg_volume_line = go.Scatter(x=data.index, y=[avg_volume] * len(data), mode='lines', name='Average Volume', line=dict(color='red', dash='dash'))

  # Create a bar chart for volume traded, color-coded based on volume status
  volume_bars = go.Bar(x=data.index, y=data['Volume'], name='Volume', marker=dict(color=data['Volume Status'].map({'Above Average': 'green', 'Below Average': 'red'})))

  # Create figure
  fig = go.Figure(data=[volume_bars, avg_volume_line])
  fig.update_layout(title='Ethereum Trading Volume', xaxis_title='Date', yaxis_title='Volume')
  
  # add date slider
  fig.update_xaxes(
    rangeslider_visible=True
  )

  # show plot
  st.plotly_chart(fig, use_container_width=True)

# Chart: ACTUAL VS PREDICTED PRICE
def plot_actual_vs_predicted(data, prediction_method):

  fig = go.Figure()
  
  fig.add_trace(go.Scatter(
    x=data['Date'], 
    y=data['Average Price (Actual)'], 
    line=dict(color='red', width=2), 
    name='Average Price (Actual)'
    ))

  fig.add_trace(go.Scatter(
    x=data['Date'], 
    y=data['Average Predicted Price'], 
    line=dict(color='green', width=2), 
    name='Average Price (Predicted)'
    ))

  fig.update_layout(
    title='Actual vs Predicted Average Price (' + prediction_method + ')', 
    xaxis_title='Date', 
    yaxis_title='Price (USD)'
    )
  
  # add date slider
  fig.update_xaxes(
    rangeslider_visible=True
  )

  # show plot
  st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------- MAIN WEB COMPONENTS ------------------------------------------- #

def main():

  # OVERRIDING YAHOO FINANCE
  yf.pdr_override()
  # px.defaults.width = 1000

  stock = "ETH-USD"

  # RETRIEVEING DATA
  ethereum_data = load_data(stock)
  tmp_df = ethereum_data.reset_index()

  # SHOWING CURREMT STOCK INFO
  metrics(ethereum_data)

  # DECLARING COLUMNS
  col1, col2 = st.columns(2, gap='small')

  # column 1
  with col1:
    
    # SHOWING HISTORICAL PRICE TREND
    plot_historical_price_trend(ethereum_data)

  # column 2
  with col2:

    # SHOWING VOLUME TRADED
    plot_trading_volume(ethereum_data)

  # SHOWING MOVING AVERAGES
  plot_moving_averages(ethereum_data)

  # SIDEBAR
  title_html = """
  <h1 style="text-align: center; font-family: 'Arial Black', sans-serif; font-size: 32px; background: linear-gradient(45deg, #FF6347, #FFD700); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">Ethereum Stock Analysis & Price Prediction App</h1>
  <hr style="border: none; background: linear-gradient(45deg, #FF6347, #FFD700); height: 2px; margin: 0 auto;"> <br> <br>
  """
  # Render the title in the sidebar
  st.sidebar.markdown(title_html, unsafe_allow_html=True)

  with st.sidebar:
    
    # TIME OF PREDICTION INPUT FIELD
    time_of_prediction = st.number_input('Time of Prediction (Days)', min_value=1)
    
    # SHORT VS LONG TERM PREDICTION METHOD RADIO
    predict_method = st.sidebar.radio( 'Select Prediction Method', (
      'Short Term', 'Long Term'
    ))

    st.button("Predict For The Next " + str(time_of_prediction) + " Days", on_click=click_button)

  # WHENEVER PREDICT BUTTON WAS CLICKED
  if st.session_state.clicked:
    if time_of_prediction == time_of_prediction:
      x = start_predict(time_of_prediction, predict_method, ethereum_data)
      # Actual vs Predicted
      plot_actual_vs_predicted(x, predict_method)
    else:
      st.warning("Already Predicted")

if __name__ == "__main__":
  main()
