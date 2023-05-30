import pandas_ta as ta
import pandas as pd

from api.enums import TechnicalIndicatorsEnum


class TechnicalIndicators:
    def __init__(self, indicators, dataframe):
        self.indicators = indicators
        self.dataframe = dataframe

    @staticmethod
    def macd(df, n_fast, n_slow):
        """Calculate MACD, MACD Signal and MACD difference
        :param df: pandas.DataFrame
        :param n_fast:
        :param n_slow:
        :return: pandas.DataFrame
        """
        EMAfast = pd.Series(df['Close'].ewm(span=n_fast, min_periods=n_slow).mean())
        EMAslow = pd.Series(df['Close'].ewm(span=n_slow, min_periods=n_slow).mean())
        MACD = pd.Series(EMAfast - EMAslow, name='MACD_' + str(n_fast) + '_' + str(n_slow))
        MACDsign = pd.Series(MACD.ewm(span=9, min_periods=9).mean(), name='MACDsign_' + str(n_fast) + '_' + str(n_slow))
        MACDdiff = pd.Series(MACD - MACDsign, name='MACDdiff_' + str(n_fast) + '_' + str(n_slow))
        df = df.join(MACD)
        df = df.join(MACDsign)
        df = df.join(MACDdiff)
        return df

    @staticmethod
    def bollinger_bands(df, n, std, add_ave=True):
        """
        :param df: pandas.DataFrame
        :param n:
        :return: pandas.DataFrame
        """

        ave = df['Close'].rolling(window=n, center=False).mean()
        sd = df['Close'].rolling(window=n, center=False).std()
        upband = pd.Series(ave + (sd * std), name='bband_upper_' + str(n))
        dnband = pd.Series(ave - (sd * std), name='bband_lower_' + str(n))
        if add_ave:
            ave = pd.Series(ave, name='bband_ave_' + str(n))
            df = df.join(pd.concat([upband, dnband, ave], axis=1))
        else:
            df = df.join(pd.concat([upband, dnband], axis=1))

        return df

    @staticmethod
    def fibonacci_retracement(df, n):
        low = df['Close'].rolling(window=n, center=False).min()
        high = df['Close'].rolling(window=n, center=False).max()
        diff = high - low

        Fib100 = pd.Series(high, name='fib_100')
        Fib764 = pd.Series(low + (diff * 0.764), name='fib_764')
        Fib618 = pd.Series(low + (diff * 0.618), name='fib_618')
        Fib50 = pd.Series(low + (diff * 0.5), name='fib_50')
        Fib382 = pd.Series(low + (diff * 0.382), name='fib_382')
        Fib236 = pd.Series(low + (diff * 0.236), name='fib_236')
        Fib0 = pd.Series(low, name='fib_0')

        df = df.join(pd.concat([Fib100, Fib764, Fib618, Fib50, Fib382, Fib236, Fib0], axis=1))

        return df

    def get_indicators_df(self):
        for indicator in self.indicators:
            if TechnicalIndicatorsEnum.has_value(indicator):
                if indicator == TechnicalIndicatorsEnum.macd.value:
                    self.dataframe = self.macd(self.dataframe, 10, 25)

                if indicator == TechnicalIndicatorsEnum.bollinger_bands.value:
                    self.dataframe = self.bollinger_bands(self.dataframe, n=20, std=4, add_ave=False)

                if indicator == TechnicalIndicatorsEnum.rsi.value:
                    self.dataframe['RSI'] = ta.rsi(self.dataframe.Close, length=15)

                if indicator == TechnicalIndicatorsEnum.ema.value:
                    self.dataframe['EMA'] = ta.ema(self.dataframe.Close, length=30)

                if indicator == TechnicalIndicatorsEnum.fibonacci.value:
                    self.dataframe = self.fibonacci_retracement(self.dataframe.Close, 30)

        return self.dataframe

