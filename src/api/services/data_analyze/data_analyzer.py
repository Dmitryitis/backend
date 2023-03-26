import pandas as pd


class DataAnalyzerOHLC:
    def __init__(self, file_url):
        self.file_url = file_url
        self.columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]

        self.dataset = pd.read_csv(file_url, delimiter=',')

    def get_head_data(self):
        return self.dataset.head(100).to_numpy()

    def get_describe(self):
        return self.dataset.describe().to_numpy()

    def get_correlation(self):
        return self.dataset.corr().to_numpy()

    def get_shape(self):
        return self.dataset.shape
