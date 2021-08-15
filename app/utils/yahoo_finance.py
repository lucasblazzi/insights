import yfinance as yf
import pandas as pd
import requests_cache


def historical(symbols, start_date, end_date):
    _symbols = " ".join(symbols)
    df = yf.download(_symbols, start=start_date, end=end_date)
    df = df[["Adjusted Close"]].droplevel(axis=1, level=0)
    return df


def simple_historical(symbols, start_date, end_date):
    adj_close = pd.DataFrame()
    session = requests_cache.CachedSession('yfinance.cache')
    session.headers['User-agent'] = 'insights/1.0'
    for symbol in symbols:
        ticker = yf.Ticker(symbol, session=session)
        hist = ticker.history(start=start_date, end=end_date)
        adj_close[symbol] = hist["Close"]
    return adj_close


def info(symbols):
    data = dict()
    for ticker in symbols:
        ticker_object = yf.Ticker(ticker)
        df = pd.DataFrame.from_dict(ticker_object.info, orient="index")
        df.reset_index(inplace=True)
        df.columns = ["Attribute", "Recent"]
        data[ticker] = df
    return data
