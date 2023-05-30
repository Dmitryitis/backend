import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from tensorflow import keras

from api.services.neural_network.neural_netwok_params import NeuralNetworkParams
from api.services.neural_network.neural_networks import ArchitectureNeuralNetworks
from api.services.techincal_indicators.technical_indicators import TechnicalIndicators


class NeuralNetworkFast:
    def __init__(self, file_url, predict_column, indicators, neural_network):
        self.file_url = file_url
        self.dataframe = pd.read_csv(file_url, delimiter=',')
        self.indicators = indicators
        self.neural_network = neural_network
        self.predict_column = predict_column
        self.training_data_split = 0.9
        self.min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        self.es = keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1,
                                                patience=20, min_delta=0.0010)
        self.fit_model = None
        self.scaled_data = None

        self.test_data = {
            'x_train_ts': [],
            'x_test_ts': [],
            'y_test_ts': [],
            'y_train_ts': []
        }

        self.predict_result = {
            'predictions': [],
            'predictions_year': [],
            'loss': [],
            'val_loss': [],
            'rmse': 0,
            'r2': 0,
            'mae': 0,
            'mape': 0,
            "rmse_train": 0,
            "r2_train": 0,
            "mape_train": 0
        }
        self.history = None

    @staticmethod
    def build_timeseries(data_scaled, time_steps,train_col, target_col):
        df = data_scaled.copy()

        x = []

        for j in range(train_col):
            x.append([])
            for i in range(time_steps, df.shape[0]):
                x[j].append(df[i - time_steps:i, j])
        x = np.moveaxis(x, [0], [2])

        x, yi = np.array(x), np.array(df[time_steps:, target_col])
        y = np.reshape(yi, (len(yi), 1))
        print(x.shape, y.shape)
        return x, y

    def prepare_data(self):
        df = self.dataframe.copy()

        techincal_indicators = TechnicalIndicators(self.indicators, df)
        df = techincal_indicators.get_indicators_df()

        df['TargetNextClose'] = df['Adj Close'].shift(-1)

        df.dropna(inplace=True)
        df.drop(['Close', 'Date'], axis=1, inplace=True)

        return df

    def fit(self):
        print(self.dataframe.columns)
        print(self.dataframe.index)

        neural_params = NeuralNetworkParams(self.neural_network)
        params = neural_params.choose_params()

        data = self.prepare_data()
        column_training = len(data.columns)

        data_scaled = self.min_max_scaler.fit_transform(data)

        x, y = self.build_timeseries(data_scaled, params['time_steps'], column_training - 1, -1)
        x_train_ts, x_test_ts, y_train_ts, y_test_ts = train_test_split(x, y, train_size=0.9, test_size=0.1,
                                                                       shuffle=False)

        self.test_data['x_train_ts'] = x_train_ts
        self.test_data['x_test_ts'] = x_test_ts
        self.test_data['y_train_ts'] = y_train_ts
        self.test_data['y_test_ts'] = y_test_ts

        networks = ArchitectureNeuralNetworks(params, column_training - 1, self.neural_network)
        model = networks.choose_model()

        history = model.fit(x_train_ts, y_train_ts,
                            verbose=2,
                            validation_data=[x_test_ts, y_test_ts],
                            batch_size=params['batch_size'],
                            epochs=params["epochs"],
                            callbacks=[self.es])

        self.fit_model = model
        self.history = history

    def predict(self):
        col_predict = -1

        x_train_ts = self.test_data['x_train_ts']
        y_train_ts = self.test_data['y_train_ts']
        x_test_ts = self.test_data['x_test_ts']
        y_test_ts = self.test_data['y_test_ts']

        y_pred = self.fit_model.predict(x_test_ts)
        y_pred_org = (y_pred * self.min_max_scaler.data_range_[col_predict]) + self.min_max_scaler.data_min_[col_predict]
        y_test_t_org = (y_test_ts * self.min_max_scaler.data_range_[col_predict]) + self.min_max_scaler.data_min_[col_predict]

        rmse = np.sqrt(np.mean(((y_pred_org - y_test_t_org) ** 2)))
        r2 = r2_score(y_test_t_org, y_pred_org)
        mae = mean_absolute_error(y_test_t_org, y_pred_org)
        mape = mean_absolute_percentage_error(y_test_t_org, y_pred_org)

        y_train_pred = self.fit_model.predict(x_train_ts)
        y_pred_train_org = (y_train_pred * self.min_max_scaler.data_range_[col_predict]) + self.min_max_scaler.data_min_[col_predict]
        y_train_ts_rescaled = (y_train_ts * self.min_max_scaler.data_range_[col_predict]) + self.min_max_scaler.data_min_[col_predict]

        rmse_train = np.sqrt(np.mean(((y_pred_train_org - y_train_ts_rescaled) ** 2)))
        r2_train = r2_score(y_train_ts_rescaled, y_pred_train_org)
        mae_train = mean_absolute_error(y_train_ts_rescaled, y_pred_train_org)
        mape_train = mean_absolute_percentage_error(y_train_ts_rescaled, y_pred_train_org)

        print(self.history.history['loss'])
        print(self.history.history['val_loss'])
        print(y_test_t_org)

        print(f"RMSE train: {rmse_train}, Rmse test: {rmse} \n")
        print(f"r2 train: {r2_train}, r2 test: {r2} \n")
        print(f"mae train: {mae_train}, mae test: {mae} \n")
        print(f"mape train: {mape_train}, mape test: {mape} \n")

        self.predict_result['predictions'] = list(y_pred_org)
        self.predict_result['rmse'] = rmse
        self.predict_result['r2'] = r2
        self.predict_result['mae'] = mae
        self.predict_result['mape'] = mape
        self.predict_result['r2_train'] = r2_train
        self.predict_result['mae_train'] = mae_train
        self.predict_result['rmse_train'] = rmse_train
        self.predict_result['mape_train'] = mape_train
        self.predict_result['loss'] = self.history.history['loss']
        self.predict_result['val_loss'] = self.history.history['val_loss']
