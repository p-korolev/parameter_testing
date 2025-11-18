import pandas as pd
import numpy as np
import yfinance as yf

class Equity:
    def __init__(self, tick: str):
        self.ticker = tick
        try:
            self.connection = yf.Ticker(tick)
        except:
            raise ValueError(f"Ticker {tick} cannot be loaded.")
    
    def history(self, start: str = None, end: str = None, period: str = '1d', interval: str = '1m') -> pd.DataFrame:
        if start!=None and end!=None:
            return self.connection.history(start=start, end=end, interval=interval)
        else:
            return self.connection.history(period=period, interval=interval)
    
    def prices(self, start: str = None, end: str = None, period: str = '1d', interval: str = '1m') -> pd.Series:
        return self.history(start=start, end=end, period=period, interval=interval)['Close']
    
    def market_cap(self) -> int:
        return self.connection.info.get("marketCap")
