# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:37:36 2024

@author: omar_
"""



import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt


#Extract the data from db to her:
    
def get_historical_data_from_db(ticker, timeframe, db_name='historical_data.db'):
    conn = sqlite3.connect(db_name)
    table_name = f"{ticker}_{timeframe}"
    query = f"SELECT * FROM {table_name} ORDER BY Date"

    try:
        # Read the data into a pandas DataFrame
        df = pd.read_sql_query(query, conn)
        
        # Convert the 'Date' column to datetime
        if timeframe == "1h" or timeframe == "4h":
            df['Date'] = df['Date'].apply(lambda x: ' '.join(x.split()[:2]))
            #df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d %H:%M:%S')
            #df['Date'] = pd.to_datetime(df['Date'].str.split().str[:2].str.join(' '), format='%Y%m%d %H:%M:%S')
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y%m%d %H:%M:%S')
            # Set the 'Date' column as the index of the DataFrame
            df.set_index('Date', inplace=True)
            
            #drop the ticker column
            df.drop('Ticker', axis=1, inplace=True)
            
            # Sort the index to ensure chronological order
            df.sort_index(inplace=True)
            
            print(f"Successfully retrieved data for {ticker} ({timeframe})")
            return df
        else:
            #df['Date'] = df['Date'].apply(lambda x: ' '.join(x.split()[:2]))
            #df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d %H:%M:%S')
            #df['Date'] = pd.to_datetime(df['Date'].str.split().str[:2].str.join(' '), format='%Y%m%d %H:%M:%S')
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y%m%d')
            # Set the 'Date' column as the index of the DataFrame
            df.set_index('Date', inplace=True)
            
            #drop the ticker column
            df.drop('Ticker', axis=1, inplace=True)
            
            # Sort the index to ensure chronological order
            df.sort_index(inplace=True)
            
            print(f"Successfully retrieved data for {ticker} ({timeframe})")
            return df

    except sqlite3.OperationalError as e:
        print(f"Error: {e}")
        print(f"Table {table_name} might not exist in the database.")
        return None

    finally:
        conn.close()
        
        
#extract between dates:
def get_historical_data_from_db_between_dates(ticker, timeframe, start_date, end_date, db_name='historical_data.db'):
    conn = sqlite3.connect(db_name)
    table_name = f"{ticker}_{timeframe}"
    query = f"""
    SELECT * FROM {table_name} 
    WHERE Date BETWEEN ? AND ?
    ORDER BY Date
    """

    try:
        # Read the data into a pandas DataFrame
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        
        if timeframe == "1h" or timeframe == "4h":
            # Convert the 'Date' column to datetime
            df['Date'] = df['Date'].apply(lambda x: ' '.join(x.split()[:2]))
            df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d %H:%M:%S')
            
            # Set the 'Date' column as the index of the DataFrame
            df.set_index('Date', inplace=True)
            
            # Drop the ticker column
            df.drop('Ticker', axis=1, inplace=True)
            
            # Sort the index to ensure chronological order
            df.sort_index(inplace=True)
            
            print(f"Successfully retrieved data for {ticker} ({timeframe}) between {start_date} and {end_date}")
            return df
        else:
            # Convert the 'Date' column to datetime
            #df['Date'] = df['Date'].apply(lambda x: ' '.join(x.split()[:2]))
            df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
            
            # Set the 'Date' column as the index of the DataFrame
            df.set_index('Date', inplace=True)
            
            # Drop the ticker column
            df.drop('Ticker', axis=1, inplace=True)
            
            # Sort the index to ensure chronological order
            df.sort_index(inplace=True)
            
            print(f"Successfully retrieved data for {ticker} ({timeframe}) between {start_date} and {end_date}")
            return df
        
   

    except sqlite3.OperationalError as e:
        print(f"Error: {e}")
        print(f"Table {table_name} might not exist in the database.")
        return None

    finally:
        conn.close()
        
#---------------------FOKUS on 4H w/ 20/60/200 SMA w/ trailing stoploss

#import the 4h 
appl_4_h = get_historical_data_from_db('AAPL', '4h')
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def prepare_data(df, SMA1, SMA2):
    df['return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['SMA1'] = df['Close'].rolling(SMA1).mean()
    df['SMA2'] = df['Close'].rolling(SMA2).mean()
    return df

def run_strategy(df, SMA1, SMA2):
    data = prepare_data(df, SMA1, SMA2).dropna()
    data['position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)
    data['strategy'] = data['position'].shift(1) * data['return']
    data.dropna(inplace=True)
    data['creturns'] = data['return'].cumsum().apply(np.exp)
    data['cstrategy'] = data['strategy'].cumsum().apply(np.exp)
    
    aperf = data['cstrategy'].iloc[-1] #absolute performance 
    operf = aperf - data['creturns'].iloc[-1] #outperformance
    return data, round(aperf, 2), round(operf, 2)

def plot_results(data, SMA1, SMA2):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['creturns'], label='Buy and Hold')
    plt.plot(data.index, data['cstrategy'], label='SMA Strategy')
    plt.title(f'SMA Strategy Performance | SMA1={SMA1}, SMA2={SMA2}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

# Example usage
SMA1, SMA2 = 20, 50

# Assuming you have a DataFrame 'df' with columns: open, high, low, Close, volume
# If you don't have the data, you can create a sample DataFrame like this:


results, aperf, operf = run_strategy(appl_4_h, SMA1, SMA2)
print(f"Strategy Performance: {aperf}")
print(f"Outperformance: {operf}")
plot_results(results, SMA1, SMA2)






