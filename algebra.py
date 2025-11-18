import math
import pandas as pd
import numpy as np

from numbers import Real
from typing import Union, List, Optional

def rolling_volatility(close_prices: pd.Series, window: Optional[int] = 21) -> pd.Series:
    '''
    Calculates annualized rolling volatility from a series of closing prices.

    :param close_prices: A pandas Series of asset closing prices indexed by date.
    :param window: The rolling window size in days used to compute standard deviation. Default is 21 (approx. one trading month).

    :return: A pd.Series of annualized rolling volatility values.

    **Examples**

    >>> import pandas as pd
    >>> import numpy as np
    >>> dates = pd.date_range(start="2024-01-01", periods=100)
    >>> prices = pd.Series(np.random.lognormal(mean=0.001, sigma=0.02, size=100), index=dates)
    >>> rolling_volatility(prices)
    2024-01-01         NaN
    2024-01-02         NaN
    ...
    2024-01-22    0.215432
    2024-01-23    0.209876
    ...
    dtype: float64
    '''
    return close_prices.pct_change().rolling(window=window).std()*np.sqrt(252)

def log(series: pd.Series) -> pd.Series:
    '''
    Applies the natural logarithm to each element in a pd.Series.

    :param series: A pandas Series of numeric values.

    :return: A pandas Series with the natural logarithm of each input value.

    **Examples**

    >>> import pandas as pd
    >>> s = pd.Series([1, np.e, 10])
    >>> log(s)
    0    0.000000
    1    1.000000
    2    2.302585
    dtype: float64
    '''
    return np.log(series)

def avg(series: pd.Series) -> np.float64:
    '''
    Returns mean of a pandas Series.

    :param series: A pandas Series of numeric values.

    :return: A single float representing the average of the input values.

    **Examples**

    >>> import pandas as pd
    >>> s = pd.Series([10, 20, 30, 40])
    >>> avg(s)
    25.0
    '''
    return series.mean()

def rolling_avg(series: pd.Series, window: Optional[int] = 7) -> pd.Series:
    '''
    Computes the rolling average over a specified window for a pandas Series.

    :param series: A pandas Series of numeric values.
    :param window: The number of periods to include in each rolling average calculation. Default is 7.

    :return: A pandas Series containing the rolling average values.

    **Examples**

    >>> import pandas as pd
    >>> s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> rolling_avg(s, window=3)
    0    NaN
    1    NaN
    2    2.0
    3    3.0
    4    4.0
    5    5.0
    6    6.0
    7    7.0
    8    8.0
    dtype: float64
    '''
    return series.rolling(window=window).mean()

def rolling_var(close_prices: pd.Series, window: Optional[int] = 21) -> pd.Series:
    '''
    Calculates rolling variance of daily returns over a specified window.

    :param close_prices: A pandas Series of asset closing prices indexed by date.
    :param window: The number of periods used to compute rolling variance. Default is 21 (approx. one trading month).

    :return: A pandas Series of rolling variance values.

    **Examples**

    >>> import pandas as pd
    >>> import numpy as np
    >>> dates = pd.date_range(start="2024-01-01", periods=30)
    >>> prices = pd.Series(np.random.lognormal(mean=0.001, sigma=0.02, size=30), index=dates)
    >>> rolling_var(prices, window=5)
    2024-01-01         NaN
    2024-01-02         NaN
    2024-01-03         NaN
    2024-01-04         NaN
    2024-01-05    0.000032
    ...
    dtype: float64
    '''
    return close_prices.pct_change().rolling(window=window).var()

def add_timerespective(first: pd.Series, second: Union[pd.Series, Real]) -> pd.Series:
    if isinstance(second, pd.Series):
        if second.index==first.index:
            return first+second
        else:
            raise ValueError("Indices do not match.")
    return first+second

def add(primary: pd.Series, secondary: pd.Series) -> pd.Series:
    '''
    Adds values from a secondary Series to a primary Series element-wise.

    :param primary: A pandas Series to be modified in-place.
    :param secondary: A pandas Series whose values will be added to the primary Series.

    :return: A pandas Series with updated values after element-wise addition.

    **Examples**

    >>> import pandas as pd
    >>> s1 = pd.Series([1, 2, 3, 4])
    >>> s2 = pd.Series([10, 20, 30])
    >>> add(s1, s2)
    0    11
    1    22
    2    33
    3     4
    dtype: int64
    '''
    secondary_values = secondary.values
    for i in range(len(secondary_values)):
        if i>len(primary):
            return primary
        primary.iloc[i] += secondary_values[i]
    return primary

def subtract_by_index(first: pd.Series, second: Union[pd.Series, Real]) -> pd.Series:
    '''
    Subtracts values from a secondary Series from a primary Series element-wise.

    :param primary: A pandas Series to be modified in-place.
    :param secondary: A pandas Series whose values will be subtracted from the primary Series.

    :return: A pandas Series with updated values after element-wise subtraction.

    **Examples**

    >>> import pandas as pd
    >>> s1 = pd.Series([10, 20, 30, 40])
    >>> s2 = pd.Series([1, 2, 3])
    >>> subtract(s1, s2)
    0     9
    1    18
    2    27
    3    40
    dtype: int64
    '''
    if isinstance(second, pd.Series):
        if second.index==first.index:
            return first-second
        else:
            raise ValueError("Indices do not match.")
    return first-second

def difference(first: pd.Series, second: Union[pd.Series, Real]) -> pd.Series:
    '''Basic first - second return.'''
    return first-second

# todo: return series
def value_signs_diff(series: pd.Series) -> List[int]:
    '''
    Computes the sign of the difference between consecutive values in a Series.

    :param series: A pandas Series of numeric values.

    :return: A list of integers where 1 indicates an increase and -1 indicates a decrease or no change.

    **Examples**

    >>> import pandas as pd
    >>> s = pd.Series([5, 7, 6, 6, 9])
    >>> value_signs_diff(s)
    [1, -1, -1, 1]
    '''
    signs = []
    vals = series.values
    for i in range(1, len(vals)):
        if vals[i]-vals[i-1]>0:
            signs.append(1)
        else:
            signs.append(-1)
    return signs

def value_signs_series(series: pd.Series) -> pd.Series:
    '''
    Returns a pandas Series indicating the sign of change between consecutive values.

    The first value is set to 0. Subsequent values are:
    - 1 if the current value is greater than the previous
    - -1 if the current value is less than or equal to the previous

    :param series: A pandas Series of numeric values.

    :return: A pandas Series of integers representing the sign of change.

    **Examples**

    >>> import pandas as pd
    >>> s = pd.Series([5, 7, 6, 6, 9])
    >>> value_signs_series(s)
    0    0
    1    1
    2   -1
    3   -1
    4    1
    dtype: int64
    '''
    diffs = series.diff()
    signs = diffs.apply(lambda x: 1 if x > 0 else -1 if x < 0 else -1)
    signs.iloc[0] = 0
    return signs.astype(int)

# todo: return series
def value_diff(series: pd.Series) -> List[int]:
    '''
    Computes the difference between consecutive values in a Series.

    :param series: A pandas Series of numeric values.

    :return: A list of integers or floats representing the change from one value to the next.

    **Examples**

    >>> import pandas as pd
    >>> s = pd.Series([10, 15, 12, 18])
    >>> value_diff(s)
    [5, -3, 6]
    '''
    diff = []
    vals = series.values 
    for i in range(1, len(vals)):
        diff.append(vals[i]-vals[i-1])
    return diff

def value_diff_series(series: pd.Series) -> pd.Series:
    '''
    Computes the difference between consecutive values in a Series and returns the result as a pandas Series.

    The first value is set to 0 to indicate no prior comparison.

    :param series: A pandas Series of numeric values.

    :return: A pandas Series of differences between consecutive values.

    **Examples**

    >>> import pandas as pd
    >>> s = pd.Series([10, 15, 12, 18])
    >>> value_diff_series(s)
    0    0
    1    5
    2   -3
    3    6
    dtype: int64
    '''
    diffs = series.diff().fillna(0)
    return diffs.astype(np.float64)

def normalize(data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    '''
    Normalizes a pandas Series or DataFrame using min-max scaling.

    Each value is scaled to a range between 0 and 1 based on its column or series minimum and maximum.

    :param data: A pandas Series or DataFrame containing numeric values.

    :return: A normalized Series or DataFrame with values scaled between 0 and 1.

    **Examples**

    >>> import pandas as pd
    >>> s = pd.Series([10, 20, 30])
    >>> normalize(s)
    0    0.0
    1    0.5
    2    1.0
    dtype: float64

    >>> df = pd.DataFrame({
    ...     'a': [1, 2, 3],
    ...     'b': [10, 20, 30]
    ... })
    >>> normalize(df)
         a    b
    0  0.0  0.0
    1  0.5  0.5
    2  1.0  1.0
    '''
    if isinstance(data, pd.Series):
        return (data-data.min())/(data.max()-data.min())
    else:
        normalized_df = data.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        return normalized_df

def scale(series: pd.Series, initial: Real = 100) -> pd.Series:
    return (series/series.iloc[0])*initial