import numpy as np
import pandas as pd


class DataAnalyzerOHLC:
    def __init__(self, file_url):
        self.file_url = file_url
        self.columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]

        self.dataset = pd.read_csv(file_url, delimiter=',')

    def set_index_datetime(self):
        self.dataset['date'] = self.dataset['Date']
        self.dataset['Date'] = pd.to_datetime(self.dataset['Date'])
        self.dataset.set_index('Date', inplace=True)

    def get_head_data(self):
        return self.dataset.head(100).to_numpy()

    def get_columns(self):
        return self.dataset.columns

    def get_describe(self):
        return self.dataset.describe().round(2).to_dict()

    def get_correlation(self):
        return self.dataset.corr().to_dict()

    def get_shape(self):
        return self.dataset.shape

    def get_quarter_profitability(self):
        quarter = self.dataset.resample('4M').mean()
        return quarter.pct_change().tail().to_numpy()

    def get_log_profitability(self):
        rets = np.log(self.dataset['Close'] / self.dataset['Close'].shift(1))

        return rets.cumsum().apply(np.exp).resample('1M').last().to_numpy()

    def get_rolling_statistics(self):
        field = 'Adj Close'
        window = 20
        rolling = pd.DataFrame()
        rolling['date'] = pd.to_datetime(self.dataset['date'])
        rolling.set_index(rolling['date'], inplace=True)

        rolling[field] = self.dataset[field]
        rolling['min'] = self.dataset[field].rolling(window=window).min()
        rolling['mean'] = self.dataset[field].rolling(window=window).mean()
        rolling['std'] = self.dataset[field].rolling(window=window).std()
        rolling['median'] = self.dataset[field].rolling(window=window).median()
        rolling['max'] = self.dataset[field].rolling(window=window).max()
        rolling['ewma'] = self.dataset[field].ewm(halflife=0.5, min_periods=window).mean()

        rolling.dropna(inplace=True)

        return {
            'time': rolling['date'].to_numpy(),
            'data_close': rolling[field].to_numpy(),
            'std': rolling['std'].to_numpy(),
            'median': rolling['median'].to_numpy(),
            'min': rolling['min'].to_numpy(),
            'mean': rolling['mean'].to_numpy(),
            'max': rolling['max'].to_numpy(),
            'ewma': rolling['ewma'].to_numpy(),
        }

    def get_sma(self):
        field = 'Adj Close'
        min_window = 42
        max_window = 252
        sma = pd.DataFrame()
        sma['date'] = pd.to_datetime(self.dataset['date'])
        sma.set_index(sma['date'], inplace=True)

        sma[field] = self.dataset[field]
        sma['SMA1'] = sma[field].rolling(window=min_window).mean()
        sma['SMA2'] = sma[field].rolling(window=max_window).mean()
        sma['position'] = np.where(sma['SMA1'] > sma['SMA2'], 1, -1)

        sma.dropna(inplace=True)

        return {
            'time': sma['date'].to_numpy(),
            'sma1': sma['SMA1'].to_numpy(),
            'sma2': sma['SMA2'].to_numpy(),
            'data_close': sma[field].to_numpy(),
            'position': sma['position'].to_numpy(),
        }