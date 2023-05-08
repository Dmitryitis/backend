import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from tensorflow import keras


class NeuralNetworkFast:
    def __init__(self, file_url, predict_column):
        self.file_url = file_url
        self.dataframe = pd.read_csv(file_url, delimiter=',')
        self.predict_column = predict_column
        self.training_data_split = 0.95
        self.min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        self.time_steps = 60
        self.es = keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1,
                                                patience=20, min_delta=0.0010)
        self.fit_model = None
        self.scaled_data = None

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
    def get_search_params():
        search_params = {'time_steps': 30, 'lstm1_nodes': 128, 'lstm1_dropout': 0.2, 'lstm2_dropout': 0.0, 'lstm2_nodes': 64, 'lr': 0.01, 'batch_size': 20, 'epochs': 30, 'optimizer': 'adam'}
        return search_params

    @staticmethod
    def get_cnn_search_params():
        search_params = {'time_steps': 60, 'lstm1_nodes': 128, 'lstm1_dropout': 0.1, 'lstm2_dropout': 0.0, 'lstm2_nodes': 64, 'lr': 0.01, 'batch_size': 30, 'epochs': 30, 'optimizer': 'adam'}

        return  search_params

    @staticmethod
    def build_timeseries(mat, y_col_index, time_steps):
        dim_0 = mat.shape[0] - time_steps
        dim_1 = mat.shape[1]
        x = np.zeros((dim_0, time_steps, dim_1))
        y = np.zeros((x.shape[0],))

        for i in range(dim_0):
            x[i] = mat[i:time_steps + i]
            y[i] = mat[time_steps + i, y_col_index]
        print("length of time-series i/o {} {}".format(x.shape, y.shape))
        return x, y

    @staticmethod
    def trim_dataset(mat, batch_size):
        """
        trims dataset to a size that's divisible by BATCH_SIZE
        """
        no_of_rows_drop = mat.shape[0] % batch_size
        if no_of_rows_drop > 0:
            return mat[:-no_of_rows_drop]
        else:
            return mat

    def prepare_data(self, search_params):
        train_cols = ["Open", "High", "Low", "Close", "Volume"]
        mat = self.dataframe.loc[:, train_cols].values

        BATCH_SIZE = search_params["batch_size"]
        TIME_STEPS = search_params["time_steps"]
        x_train, x_test = train_test_split(mat, train_size=0.8, test_size=0.2, shuffle=False)

        # scale the train and test dataset
        x_train = self.min_max_scaler.fit_transform(x_train)
        x_test = self.min_max_scaler.transform(x_test)

        x_train_ts, y_train_ts = self.build_timeseries(x_train, 3, TIME_STEPS)
        x_test_ts, y_test_ts = self.build_timeseries(x_test, 3, TIME_STEPS)
        x_train_ts = self.trim_dataset(x_train_ts, BATCH_SIZE)
        y_train_ts = self.trim_dataset(y_train_ts, BATCH_SIZE)
        print("Train size(trimmed) {}, {}".format(x_train_ts.shape, y_train_ts.shape))
        # this is to check if formatting of data is correct
        print("{},{}".format(x_train[TIME_STEPS - 1, 3], y_train_ts[0]))
        print(str(x_train[TIME_STEPS, 3]), str(y_train_ts[1]))
        print(str(x_train[TIME_STEPS + 1, 3]), str(y_train_ts[2]))
        print(str(x_train[TIME_STEPS + 2, 3]), str(y_train_ts[3]))
        print(str(x_train[TIME_STEPS + 3, 3]), str(y_train_ts[4]))
        print(str(x_train[TIME_STEPS + 4, 3]), str(y_train_ts[5]))
        print(str(x_train[TIME_STEPS + 5, 3]), str(y_train_ts[6]))
        x_test_ts = self.trim_dataset(x_test_ts, BATCH_SIZE)
        y_test_ts = self.trim_dataset(y_test_ts, BATCH_SIZE)

        return x_train_ts, y_train_ts, x_test_ts, y_test_ts

    def create_model_cnn(self):

        params = self.get_cnn_search_params()

        model = keras.Sequential()
        model.add(keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(params['time_steps'], 5)))
        model.add(keras.layers.MaxPool1D(2))
        model.add(keras.layers.LSTM(params["lstm1_nodes"],
                                    dropout=0.0,
                                    recurrent_dropout=0.0,
                                    stateful=False,
                                    recurrent_regularizer=keras.regularizers.L1(0.01),
                                    return_sequences=True,
                                    input_shape=(params["time_steps"], 5)))
        model.add(keras.layers.LSTM(params["lstm2_nodes"], dropout=params["lstm2_dropout"], return_sequences=False))
        model.add(keras.layers.Dense(1))

        lr = params["lr"]
        epochs = params["epochs"]

        if params["optimizer"] == 'rms':
            optimizer = keras.optimizers.RMSprop(learning_rate=lr)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=lr)

        # Compile the model
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    def create_model(self):
        # Build the LSTM model
        params = self.get_search_params()

        model = keras.Sequential()
        model.add(keras.layers.LSTM(params["lstm1_nodes"],
                                    dropout=0.0,
                                    recurrent_dropout=0.0,
                                    stateful=False,
                                    recurrent_regularizer=keras.regularizers.L1(0.01),
                                    return_sequences=True,
                                    input_shape=(params["time_steps"], 5)))
        model.add(keras.layers.LSTM(params["lstm2_nodes"], dropout=params["lstm2_dropout"], return_sequences=False))
        model.add(keras.layers.Dense(1))

        lr = params["lr"]
        epochs = params["epochs"]

        if params["optimizer"] == 'rms':
            optimizer = keras.optimizers.RMSprop(learning_rate=lr)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=lr)

        # Compile the model
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    def fit(self):
        if self.dataframe.shape[0] > 5000:
            params = self.get_cnn_search_params()
            epochs = params["epochs"]
            x_train_ts, y_train_ts, x_test_ts, y_test_ts = self.prepare_data(params)
            model = self.create_model_cnn()
        else:
            params = self.get_search_params()
            epochs = params["epochs"]
            x_train_ts, y_train_ts, x_test_ts, y_test_ts = self.prepare_data(params)
            model = self.create_model()

        history = model.fit(x_train_ts, y_train_ts,
                            verbose=2,
                            validation_data=[x_test_ts, y_test_ts],
                            batch_size=params['batch_size'],
                            epochs=epochs,
                            callbacks=[self.es])

        self.fit_model = model
        self.history = history

    def predict_year(self, x_test_ts):
        X_FUTURE = 365
        predictions = np.array([])
        last = x_test_ts[-1]
        print(last)
        for i in range(X_FUTURE):
            curr_prediction = self.fit_model.predict(np.array([last]))
            last = np.concatenate([last[1:], curr_prediction])
            predictions = np.concatenate([predictions, curr_prediction[0]])
        predictions = self.min_max_scaler.inverse_transform([predictions])[0]
        print(predictions)

        return predictions

    def predict(self):
        # Get the models predicted price values
        if self.dataframe.shape[0] > 5000:
            search_params = self.get_cnn_search_params()
        else:
            search_params = self.get_search_params()
        x_train_ts, y_train_ts, x_test_ts, y_test_ts = self.prepare_data(search_params)

        y_pred = self.fit_model.predict(x_test_ts)
        y_pred_org = (y_pred * self.min_max_scaler.data_range_[3]) + self.min_max_scaler.data_min_[3]
        y_test_t_org = (y_test_ts * self.min_max_scaler.data_range_[3]) + self.min_max_scaler.data_min_[3]

        rmse = np.sqrt(np.mean(((y_pred_org - y_test_t_org) ** 2)))
        r2 = r2_score(y_test_t_org, y_pred_org)
        mae = mean_absolute_error(y_test_t_org, y_pred_org)
        mape = mean_absolute_percentage_error(y_test_t_org, y_pred_org)

        y_train_pred = self.fit_model.predict(x_train_ts)
        y_pred_train_org = (y_train_pred * self.min_max_scaler.data_range_[3]) + self.min_max_scaler.data_min_[3]
        y_train_ts_rescaled = (y_train_ts * self.min_max_scaler.data_range_[3]) + self.min_max_scaler.data_min_[3]

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
