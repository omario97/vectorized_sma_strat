import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf

# Function to get data from yfinance
def get_historical_data_from_yf(ticker, timeframe, period):
    if timeframe == "1h":
        interval = "1h"
    else:  # 1d
        interval = "1d"
   
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        df.index = df.index.strftime('%Y%m%d %H:%M')
        return df
    except Exception as e:
        st.error(f"Error: {e}. Failed to retrieve data for {ticker}.")
        return None

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
st.set_page_config(layout="wide")

# Sidebar
st.sidebar.markdown("""
# Links
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/omar-hussain-504777164/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/omario97)
""")

st.sidebar.title('Strategy Parameters')

# User inputs in sidebar
ticker = st.sidebar.selectbox('Select Ticker', ["AAPL", "MSFT", "AMD", "TSLA", "ROKU", "NVDA", "OXY"])
timeframe = st.sidebar.selectbox('Select Timeframe', ['1h', '1d'])
period = st.sidebar.selectbox('Select Period', ['1mo', '3mo', '6mo', 'ytd','1y', '2y'])
SMA1 = st.sidebar.number_input('SMA1', min_value=1, max_value=200, value=20)
SMA2 = st.sidebar.number_input('SMA2', min_value=1, max_value=200, value=50)

# Main content
st.title('SMA Strategy Backtester')

# Explanation
st.write("""
## How it works
This tool implements a Simple Moving Average (SMA) crossover strategy and compares its performance to a buy-and-hold approach.

The complete source code for this application can be found on the GitHub page linked in the sidebar.         

### The Strategy
1. Calculate two SMAs: a short-term (SMA1) and a long-term (SMA2).
2. Buy (or stay long) when SMA1 > SMA2.
3. Sell (or stay out) when SMA1 < SMA2.

### Calculations
- **Strategy Performance**: The final value of $1 invested using the SMA strategy.
- **Outperformance**: The difference between the SMA strategy's final value and the buy-and-hold strategy's final value.

### Limitations
This is a simplified, vectorized backtest which does not account for:
- Trading costs and slippage
- Liquidity constraints
- Survivorship bias
- Look-ahead bias

Real-world performance may differ significantly from these results.
""")

if st.sidebar.button('Run Strategy'):
    # Get data
    df = get_historical_data_from_yf(ticker, timeframe, period)
   
    if df is not None and len(df) > max(SMA1, SMA2):
        # Run strategy
        results, aperf, operf = run_strategy(df, SMA1, SMA2)
       
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Strategy Performance", f"{aperf:.2f}")
        with col2:
            st.metric("Outperformance vs Buy & Hold", f"{operf:.2f}")
       
        # Plot results
        fig = plot_results(results, SMA1, SMA2)
        st.plotly_chart(fig, use_container_width=True)
    else:
        if df is None:
            st.error("Failed to retrieve data. Please check your inputs and try again.")
        else:
            st.error(f"Not enough data for the selected period and SMAs. Please choose a longer period or smaller SMA values. Available data points: {len(df)}")
else:
    st.write("Please select your parameters and click 'Run Strategy' to see the results.")