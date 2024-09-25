import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sqlite3

# Function to get data from SQLite database
def get_historical_data_from_db(ticker, timeframe, db_name='historical_data.db'):
    conn = sqlite3.connect(db_name)
    table_name = f"{ticker}_{timeframe}"
    query = f"SELECT * FROM {table_name} ORDER BY Date"

    try:
        df = pd.read_sql_query(query, conn)
        
        if timeframe == "1h" or timeframe == "4h":
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y%m%d %H:%M:%S')
        else:
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y%m%d')
        
        df.set_index('Date', inplace=True)
        df.drop('Ticker', axis=1, inplace=True)
        df.sort_index(inplace=True)
        
        return df

    except sqlite3.OperationalError as e:
        st.error(f"Error: {e}. Table {table_name} might not exist in the database.")
        return None

    finally:
        conn.close()

# Function to prepare data
def prepare_data(df, SMA1, SMA2):
    df['return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['SMA1'] = df['Close'].rolling(SMA1).mean()
    df['SMA2'] = df['Close'].rolling(SMA2).mean()
    return df

# Function to run strategy
def run_strategy(df, SMA1, SMA2):
    data = prepare_data(df, SMA1, SMA2).dropna()
    data['position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)
    data['strategy'] = data['position'].shift(1) * data['return']
    data.dropna(inplace=True)
    data['creturns'] = data['return'].cumsum().apply(np.exp)
    data['cstrategy'] = data['strategy'].cumsum().apply(np.exp)
    
    aperf = data['cstrategy'].iloc[-1]
    operf = aperf - data['creturns'].iloc[-1]
    return data, round(aperf, 2), round(operf, 2)

# Function to plot results using Plotly
def plot_results(data, SMA1, SMA2):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['creturns'], mode='lines', name='Buy and Hold'))
    fig.add_trace(go.Scatter(x=data.index, y=data['cstrategy'], mode='lines', name='SMA Strategy'))
    
    fig.update_layout(
        title=f'SMA Strategy Performance | SMA1={SMA1}, SMA2={SMA2}',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        legend_title='Strategy',
        hovermode="x unified"
    )
    
    return fig

# Streamlit app
st.title('SMA Strategy Backtester')

# User inputs
ticker = st.selectbox('Select Ticker', ["AAPL", "MSFT","AMD","TSLA","ROKU","NVDA","OXY"])  # Add more tickers as needed
timeframe = st.selectbox('Select Timeframe', ['1h', '4h', '1d'])
SMA1 = st.number_input('SMA1', min_value=1, max_value=200, value=20)
SMA2 = st.number_input('SMA2', min_value=1, max_value=200, value=50)

if st.button('Run Strategy'):
    # Get data
    df = get_historical_data_from_db(ticker, timeframe)
    
    if df is not None:
        # Run strategy
        results, aperf, operf = run_strategy(df, SMA1, SMA2)
        
        # Display results
        st.write(f"Strategy Performance: {aperf}")
        st.write(f"Outperformance: {operf}")
        
        # Plot results
        fig = plot_results(results, SMA1, SMA2)
        st.plotly_chart(fig)
    else:
        st.error("Failed to retrieve data. Please check your database and inputs.")