import numpy as np
import optuna
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from tensorflow import keras
from sklearn.model_selection import train_test_split


class StopWhenTrialKeepBeingPrunedCallback:
    def __init__(self, threshold: int):
        self.threshold = threshold
        self._consequtive_pruned_count = 0

    def __call__(self, study, trial):
        if trial.state == optuna.trial.TrialState.PRUNED:
            self._consequtive_pruned_count += 1
        else:
            self._consequtive_pruned_count = 0

        if self._consequtive_pruned_count >= self.threshold:
            study.stop()


class NeuralNetworkLstm:
    def __init__(self, file_url, predict_column):
        self.file_url = file_url
        self.dataframe = pd.read_csv(file_url, delimiter=',')
        self.predict_column = predict_column
        self.training_data_split = 0.95
        self.min_max_scaler = MinMaxScaler()
        self.time_steps = 60
        self.es = keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1,
                                                patience=20, min_delta=0.0010)
        self.fit_model = None
        self.scaled_data = None

        self.predict_result = {
            'predictions': [],
            'rmse': 0,
            'r2': 0,
            'mae': 0,
            'mape': 0
        }

    @staticmethod
    def trial_search_params(trial):
        search_params = {
            'time_steps': trial.suggest_int('ts', 30, 60, 30),
            'lstm1_nodes': trial.suggest_int('lstm1_nodes', 100, 128, 28),
            'lstm1_dropout': trial.suggest_float('lstm1_dropout', 0.1, 0.2, step=0.1),
            'lstm2_dropout': trial.suggest_float('lstm2_dropout', 0.0, 0.2, step=0.1),
            'lstm2_nodes': trial.suggest_int('lstm2_nodes', 64, 64, 64),
            "lr": trial.suggest_float('lr', 0.01, 0.01),
            "batch_size": trial.suggest_int('batch_size', 20, 40, 10),
            "epochs": trial.suggest_int('epochs', 30, 60, 30),
            "optimizer": trial.suggest_categorical('optimizer', ['adam', 'rms'])
        }

        return search_params

    @staticmethod
    def trial_bi_search_params(trial):
        search_params = {
            'time_steps': trial.suggest_int('ts', 30, 60, 30),
            'lstm1_nodes': trial.suggest_int('lstm1_nodes', 100, 128, 28),
            'lstm1_dropout': trial.suggest_float('lstm1_dropout', 0.1, 0.2, step=0.1),
            'lstm2_dropout': trial.suggest_float('lstm2_dropout', 0.0, 0.2, step=0.1),
            'lstm2_nodes': trial.suggest_int('lstm2_nodes', 64, 64, 64),
            "lr": trial.suggest_float('lr', 0.01, 0.01),
            "batch_size": trial.suggest_int('batch_size', 20, 40, 10),
            "epochs": trial.suggest_int('epochs', 30, 30, 30),
            "optimizer": trial.suggest_categorical('optimizer', ['rms'])
        }

        return search_params

    @staticmethod
    def trial_cnn_search_params(trial):
        search_params = {
            'time_steps': trial.suggest_int('ts', 30, 60, 30),
            'lstm1_nodes': trial.suggest_int('lstm1_nodes', 100, 128, 28),
            'lstm1_dropout': trial.suggest_float('lstm1_dropout', 0.1, 0.2, step=0.1),
            'lstm2_dropout': trial.suggest_float('lstm2_dropout', 0.0, 0.2, step=0.1),
            'lstm2_nodes': trial.suggest_int('lstm2_nodes', 64, 64, 64),
            "lr": trial.suggest_float('lr', 0.01, 0.01),
            "batch_size": trial.suggest_int('batch_size', 20, 40, 10),
            "epochs": trial.suggest_int('epochs', 30, 30, 30),
            "optimizer": trial.suggest_categorical('optimizer', ['rms'])
        }

        return search_params

    @staticmethod
    def get_search_params():
        search_params = {
            "lstm_layers": [1, 2],
            "dense_layers": [1, 2],
            "lstm1_nodes": 128,
            "lstm2_nodes": 64,
            "dense2_nodes": 20,
            "batch_size": 20,
            "time_steps": 60,
            "lr": 0.01,
            "epochs": 30,
            "optimizer": 'rms'
        }

        return search_params

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

        BATCH_SIZE = search_params["batch_size"]  # {{choice([20, 30, 40, 50])}}
        TIME_STEPS = search_params["time_steps"]  # {{choice([30, 60, 90])}}
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

    def model_fit_result(self, model, x_test_ts, x_train_ts, y_test_ts, y_train_ts):
        y_pred = model.predict(x_test_ts)
        y_pred_org = (y_pred * self.min_max_scaler.data_range_[3]) + self.min_max_scaler.data_min_[3]
        y_test_t_org = (y_test_ts * self.min_max_scaler.data_range_[3]) + self.min_max_scaler.data_min_[3]

        y_train_pred = model.predict(x_train_ts)
        y_pred_train_org = (y_train_pred * self.min_max_scaler.data_range_[3]) + self.min_max_scaler.data_min_[3]
        y_train_ts_rescaled = (y_train_ts * self.min_max_scaler.data_range_[3]) + self.min_max_scaler.data_min_[3]

        rmse = np.sqrt(np.mean(((y_pred_org - y_test_t_org) ** 2)))
        r2_train = r2_score(y_train_ts_rescaled, y_pred_train_org)
        r2 = r2_score(y_test_t_org, y_pred_org)

        return rmse, r2, r2_train

    def predict(self):
        # Get the models predicted price values
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

        print(y_pred_org)
        print(y_test_t_org)

        print(f"RMSE train: {rmse_train}, Rmse test: {rmse} \n")
        print(f"r2 train: {r2_train}, r2 test: {r2} \n")
        print(f"mae train: {mae_train}, mae test: {mae} \n")
        print(f"mape train: {mape_train}, mape test: {mape} \n")

        self.predict_result['predictions'] = list(y_pred_org)
        # self.predict_result['rmse'] = rmse
        # self.predict_result['r2'] = r2
        # self.predict_result['mae'] = mae
        # self.predict_result['mape'] = mape

    def create_model_bi_lstm(self, trial):
        # Build the LSTM model

        params = self.trial_bi_search_params(trial)

        x_train_ts, y_train_ts, x_test_ts, y_test_ts = self.prepare_data(params)

        model = keras.Sequential()

        forward_layer = keras.layers.LSTM(params["lstm1_nodes"],
                                    dropout=params["lstm1_dropout"],
                                    recurrent_dropout=0.2,
                                    stateful=False,
                                    recurrent_regularizer=keras.regularizers.L1(0.01),
                                    return_sequences=True,
                                    kernel_initializer='random_uniform')
        backward_layer = keras.layers.LSTM(params["lstm1_nodes"],
                                    dropout=params["lstm1_dropout"],
                                    recurrent_dropout=0.2,
                                    stateful=False,
                                    recurrent_regularizer=keras.regularizers.L1(0.01),
                                    return_sequences=True,
                                    kernel_initializer='random_uniform',
                                           go_backwards=True)

        model.add(keras.layers.Bidirectional(forward_layer,input_shape=(params['time_steps'], 5), backward_layer=backward_layer,))
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

        history = model.fit(x_train_ts, y_train_ts,
                            verbose=2,
                            validation_data=[x_test_ts, y_test_ts],
                            batch_size=params['batch_size'],
                            epochs=epochs,
                            callbacks=[self.es])

        rmse, r2, r2_train = self.model_fit_result(model, x_test_ts, x_train_ts, y_test_ts, y_train_ts)

        self.fit_model = model

        print(f'Best validation error of epoch: {rmse}, {r2}, train: {r2_train}')

        return np.abs(r2_train - r2)


    def create_model_cnn_lstm(self, trial):
        params = self.trial_cnn_search_params(trial)

        x_train_ts, y_train_ts, x_test_ts, y_test_ts = self.prepare_data(params)

        model = keras.Sequential()
        model.add(keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(params['time_steps'], 5)))
        model.add(keras.layers.MaxPool1D(2))
        model.add(keras.layers.LSTM(params["lstm1_nodes"],
                                    dropout=params["lstm1_dropout"],
                                    recurrent_dropout=0.2,
                                    stateful=False,
                                    recurrent_regularizer=keras.regularizers.L1(0.01),
                                    return_sequences=True,
                                    input_shape=(params["time_steps"], 5),
                                    kernel_initializer='random_uniform'))
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

        history = model.fit(x_train_ts, y_train_ts,
                            verbose=2,
                            validation_data=[x_test_ts, y_test_ts],
                            batch_size=params['batch_size'],
                            epochs=epochs,
                            callbacks=[self.es])

        rmse, r2, r2_train = self.model_fit_result(model, x_test_ts, x_train_ts, y_test_ts, y_train_ts)

        self.fit_model = model

        print(f'Best validation error of epoch: {rmse}, {r2}, train: {r2_train}')

        return np.abs(r2_train - r2)

    def create_model(self, trial):
        # Build the LSTM model

        params = self.trial_search_params(trial)

        x_train_ts, y_train_ts, x_test_ts, y_test_ts = self.prepare_data(params)

        model = keras.Sequential()
        model.add(keras.layers.LSTM(params["lstm1_nodes"],
                                    dropout=params["lstm1_dropout"],
                                    recurrent_dropout=0.2,
                                    stateful=False,
                                    recurrent_regularizer=keras.regularizers.L1(0.01),
                                    return_sequences=True,
                                    input_shape=(params["time_steps"], 5),
                                    kernel_initializer='random_uniform'))
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

        history = model.fit(x_train_ts, y_train_ts,
                            verbose=2,
                            validation_data=[x_test_ts, y_test_ts],
                            batch_size=params['batch_size'],
                            epochs=epochs,
                            callbacks=[self.es])

        rmse, r2, r2_train = self.model_fit_result(model, x_test_ts, x_train_ts, y_test_ts, y_train_ts)

        self.fit_model = model

        print(f'Best validation error of epoch: {rmse}, {r2}, train: {r2_train}')

        return np.abs(r2_train - r2)

    def fit(self):
        # {'ts': 60, 'lstm1_nodes': 128, 'lstm2_nodes':
        # 64, 'lr': 0.01, 'batch_size': 30, 'epochs': 60, 'optimizer': 'rms'}
        #  {'ts': 60, 'lstm1_nodes': 100, 'lstm1_dropout'
        # : 0.17578316432879248, 'lstm2_nodes': 64, 'lr': 0.01, 'batch_size': 30, 'epochs': 60, 'optimizer': 'rms'} - best

        #  {'ts': 60, 'lstm1_nodes': 100, 'lstm1_dropout
        # ': 0.19979944936110774, 'lstm2_nodes': 64, 'lr': 0.01, 'batch_size': 40, 'epochs': 30, 'optimizer': 'rms'}

        # {'ts': 60, 'lstm1_nodes': 128, 'lstm1_dropout':
        #  0.1743614685412185, 'lstm2_nodes': 64, 'lr': 0.01, 'batch_size': 40, 'epochs': 30, 'optimizer': 'rms'}.

        #  {'ts': 30, 'lstm1_nodes': 100, 'lstm1_dropout
        # ': 0.1, 'lstm2_dropout': 0.1, 'lstm2_nodes': 64, 'lr': 0.01, 'batch_size': 20, 'epochs': 30, 'optimizer': 'adam'}

        #  {'ts': 30, 'lstm1_nodes': 128, 'lstm1_dropou
        # t': 0.2, 'lstm2_dropout': 0.0, 'lstm2_nodes': 64, 'lr': 0.01, 'batch_size': 20, 'epochs': 30, 'optimizer': 'adam'}. - l1

        #  {'ts': 60, 'lstm1_nodes': 128, 'lstm1_dropou
        # t': 0.1, 'lstm2_dropout': 0.0, 'lstm2_nodes': 64, 'lr': 0.01, 'batch_size': 30, 'epochs': 30, 'optimizer': 'adam'} - l1cnn

        #  {'ts': 60, 'lstm1_nodes': 128, 'lstm1_dropo
        # ut': 0.2, 'lstm2_dropout': 0.0, 'lstm2_nodes': 64, 'lr': 0.01, 'batch_size': 20, 'epochs': 30, 'optimizer': 'rms'} - l1cnn big data

        #  {'ts': 60, 'lstm1_nodes': 100, 'lstm1_dropou
        # t': 0.2, 'lstm2_dropout': 0.2, 'lstm2_nodes': 64, 'lr': 0.01, 'batch_size': 20, 'epochs': 30, 'optimizer': 'rms'} - l1cnn big data
        study_stop_cb = StopWhenTrialKeepBeingPrunedCallback(2)
        study = optuna.create_study(pruner=optuna.pruners.HyperbandPruner())
        study.optimize(self.create_model_bi_lstm, n_trials=10, callbacks=[study_stop_cb])

        print(study.best_trial.value)
