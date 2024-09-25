# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 08:25:17 2024

@author: omar_
"""

'''

Goal: donwload, store and import data and perform backtesting via a bactesting engine.


'''


from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import pandas as pd
import threading
import time
from queue import Queue
import os
import sqlite3
import numpy as np
import pytz

class IBConnection(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False
        self.next_order_id = None
        self.orders = {}
        self.historical_data = {}
        self.req_id_counter = 1
        self.data_queue = Queue()

    # Connection related methods
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        print("Error {} {} {} {}".format(reqId, errorCode, errorString, advancedOrderRejectJson))

    def connectAck(self):
        self.connected = True
        print("Connected to IB API")

    def connect_to_ib(self):
        self.connect('127.0.0.1', 7497, clientId=23)
        api_thread = threading.Thread(target=self.run, daemon=True)
        api_thread.start()
        while not self.connected:
            time.sleep(0.1)

    def disconnect_from_ib(self):
        self.disconnect()
        print("Disconnected from IB API")

    # Order related methods
    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.next_order_id = orderId
        print(f"Next Valid Order ID: {orderId}")
        
        
    def get_next_req_id(self):
        req_id = self.req_id_counter
        self.req_id_counter += 1
        return req_id
        
        
#Historical data ----------------
        
    def historicalData(self, reqId, bar):
        if reqId not in self.historical_data:
            self.historical_data[reqId] = []
        self.historical_data[reqId].append({
            "Date": bar.date,
            "Open": bar.open,
            "High": bar.high,
            "Low": bar.low,
            "Close": bar.close,
            "Volume": bar.volume
        })
        

    def historicalDataEnd(self, reqId, start, end):
        print(f"Historical data end for reqId: {reqId}")
        self.data_queue.put(reqId)

#The eclient function
    def request_historical_data(self, reqId, contract, endDateTime, durationStr, barSizeSetting):
        self.reqHistoricalData(
            reqId=reqId,
            contract=contract,
            endDateTime=endDateTime,
            durationStr=durationStr,
            barSizeSetting=barSizeSetting,
            whatToShow='ADJUSTED_LAST',
            useRTH=1,
            formatDate=1,
            keepUpToDate=0,
            chartOptions=[]
        )
        
def create_stock_contract(symbol, exchange="SMART", currency="USD"):
    contract = Contract()
    contract.symbol = symbol
    contract.secType = "STK"
    contract.exchange = exchange
    contract.currency = currency
    return contract


# The main function
def get_historical_data(app, tickers, timeframes):
    result = {}
    for ticker in tickers:
        result[ticker] = {}
        contract = create_stock_contract(ticker)
        
        for tf in timeframes:
            reqId = app.get_next_req_id()
            if tf == "1h":
                app.request_historical_data(reqId, contract, '', "5 Y", "1 hour")
            elif tf == "4h":
                app.request_historical_data(reqId, contract, '', "5 Y", "4 hours")
            elif tf == "1d":
                app.request_historical_data(reqId, contract, '', "5 Y", "1 day")
            
            # Wait for this request to complete
            completed_req_id = app.data_queue.get() #important
            if completed_req_id == reqId:
                result[ticker][tf] = pd.DataFrame(app.historical_data[reqId])
                result[ticker][tf].set_index('Date', inplace=True)
                print(f"Completed historical data request for {ticker} - {tf}")
            else:
                print(f"Unexpected reqId received: {completed_req_id}")
            
            time.sleep(2)  # Sleep to avoid pacing violations

    return result



#THe database

def create_and_update_database(historical_data, db_name='historical_data.db'):
    db_exists = os.path.exists(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    for ticker, timeframes in historical_data.items():
        for tf, df in timeframes.items():
            table_name = f"{ticker}_{tf}"

            # Check if 'Date' is in the index
            if 'Date' not in df.columns and df.index.name == 'Date':
                df = df.reset_index()

            # Ensure data types are compatible with SQLite
            df = df.astype({
                'Open': float,
                'High': float,
                'Low': float,
                'Close': float,
                'Volume': np.int64  # Use int64 to avoid overflow
            })

            # Convert Date to string if it's not already
            if 'Date' in df.columns and not pd.api.types.is_string_dtype(df['Date']):
                df['Date'] = df['Date'].astype(str)

            # Add Ticker column
            df['Ticker'] = ticker

            # Check if the table already exists
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            table_exists = cursor.fetchone() is not None

            if not table_exists:
                # Create the table if it doesn't exist
                cursor.execute(f'''CREATE TABLE {table_name}
                                   (Ticker TEXT,
                                    Date TEXT,
                                    Open REAL,
                                    High REAL,
                                    Low REAL,
                                    Close REAL,
                                    Volume INTEGER,
                                    PRIMARY KEY (Ticker, Date))''')
                print(f"Created table: {table_name}")

                # Insert all data into the new table
                df.to_sql(table_name, conn, if_exists='append', index=False)
                print(f"Inserted data into new table: {table_name}")
            else:
                # Table exists, so we need to append only new data
                # First, get the latest date in the existing table
                cursor.execute(f"SELECT MAX(Date) FROM {table_name}")
                latest_date = cursor.fetchone()[0]

                # Filter the dataframe to only include dates after the latest date in the database
                new_data = df[df['Date'] > latest_date]

                if not new_data.empty:
                    # Append only the new data
                    new_data.to_sql(table_name, conn, if_exists='append', index=False)
                    print(f"Appended {len(new_data)} new rows to table: {table_name}")
                else:
                    print(f"No new data to append for table: {table_name}")

    # Create index on (Ticker, Date) for each table
    for table in pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)['name']:
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_ticker_date ON {table} (Ticker, Date)")

    conn.commit()
    conn.close()
    print(f"Database {'updated' if db_exists else 'created'}: {db_name}")

    # Print data types for debugging
    for ticker, timeframes in historical_data.items():
        for tf, df in timeframes.items():
            print(f"\nData types for {ticker}_{tf}:")
            print(df.dtypes)
            print("\nNull value count:")
            print(df.isnull().sum())
    




#import the db data into a dataframe:



def get_historical_data_from_db(ticker, timeframe, db_name='historical_data.db'):
    conn = sqlite3.connect(db_name)
    table_name = f"{ticker}_{timeframe}"
    query = f"SELECT * FROM {table_name} ORDER BY Date"

    try:
        # Read the data into a pandas DataFrame
        df = pd.read_sql_query(query, conn)
        
        # Convert the 'Date' column to datetime
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

    except sqlite3.OperationalError as e:
        print(f"Error: {e}")
        print(f"Table {table_name} might not exist in the database.")
        return None

    finally:
        conn.close()





if __name__ == "__main__":
    
    directory = r"C:\Users\omar_\OneDrive\Skrivebord\TWS_algo_strat"
    os.chdir(directory)
    
    app = IBConnection()
    app.connect_to_ib()
    time.sleep(1)  # Wait for connection

    tickers = ["AAPL", "MSFT","AMD","TSLA","ROKU","NVDA","OXY"]
    timeframes = ["1h", "4h", "1d"] #mapping

    historical_data = get_historical_data(app, tickers, timeframes)

    # Print sample of the data
    for ticker in tickers:
        for tf in timeframes:
            print(f"{ticker} - {tf}:")
            print(historical_data[ticker][tf].head())
            print("\n")

    app.disconnect_from_ib()
    
    create_and_update_database(historical_data)
    
    
    #Testing of importing the data
    # appl_1_h = get_historical_data_from_db('AAPL', '1h')


    
    # aapl_1h_data = start_date = '20200101 00:00:00'  # January 1, 2020
    # end_date = '20201231 23:59:59'    # December 31, 2020
    # aapl_1h_data = get_historical_data_from_db_between_dates('AAPL', '1h', start_date, end_date)

    

    