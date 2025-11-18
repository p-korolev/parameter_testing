import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sci
import requests

from functools import lru_cache
from typing import Union, List
from numbers import Real
from io import StringIO

from equity import Equity
from algebra import scale

class SectorGroup:
    def __init__(self, sector_name: str = None):
        if sector_name==None:
            raise ValueError("sector_name must be provided.")
        self.sec = sector_name.lower()
        self.group_size = 0
        self.tickers = []
        self.caps = []

    @lru_cache
    def load_group(self, count: int = 100) -> None:
        url = f"https://finance.yahoo.com/research-hub/screener/sec-ind_sec-largest-equities_{self.sec}/?start=0&count={count}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # will raise HTTPError if not 200
        tables = pd.read_html(StringIO(response.text), index_col=0)
        # get tickers and clean
        symbols = tables[0]["Symbol"].dropna().values
        for i in range(len(symbols)):
            if " " in symbols[i]:
                spl = symbols[i].split(' ')
                symbols[i] = spl[1]

        self.tickers = symbols.tolist()
        self.group_size = len(self.tickers)
    
    @lru_cache
    def load_caps(self) -> None:
        if self.group_size == 0:
            raise IndexError("Sector group has not been loaded.")
        else:
            for e in self.tickers:
                try:
                    self.caps.append(Equity(tick=e).market_cap())
                except:
                    self.caps.append(None)
    
    def get_scaled_caps(self) -> List[float]:
        if len(self.caps)==0:
            return None
        minlen = len(str(self.caps[-1]).strip())
        scaled = []
        for cap in self.caps:
            if isinstance(cap, int):
                scaled.append(cap/(10**(minlen)))
        return scaled

    def mean_cap(self, scale: float = 1.0) -> Real:
        return np.mean(self.caps)/scale


        
    
