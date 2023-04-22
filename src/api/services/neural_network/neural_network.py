import numpy as np
import pandas as pd
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
                                patience=40, min_delta=0.0010)
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
    def prepare_train_data(training_data_len, scaled_data, time_steps):
        train_data = scaled_data[0:int(training_data_len), :]
        # Split the data into x_train and y_train data sets
        x_train = []
        y_train = []

        for i in range(time_steps, len(train_data)):
            x_train.append(train_data[i - time_steps:i, 0])
            y_train.append(train_data[i, 0])
            if i <= time_steps + 1:
                print(x_train)
                print(y_train)
                print()

        # Convert the x_train and y_train to numpy arrays
        x_train, y_train = np.array(x_train), np.array(y_train)

        # Reshape the data
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        return x_train, y_train

    @staticmethod
    def prepare_test_data(training_data_len, test_data,dataset, time_steps):
        # Create the data sets x_test and y_test
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(time_steps, len(test_data)):
            x_test.append(test_data[i - time_steps:i, 0])

        # Convert the data to a numpy array
        x_test = np.array(x_test)

        # Reshape the data
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        return x_test, y_test

    @staticmethod
    def training_data_len(len_dataset, training_data_split):
        return int(np.ceil(len_dataset * training_data_split))

    @staticmethod
    def prepare_data(dataframe, column_name):
        data = dataframe.filter([column_name])
        dataset = data.values
        return dataset

    @staticmethod
    def create_model(x_train):
        # Build the LSTM model
        model = keras.Sequential()
        model.add(keras.layers.LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(keras.layers.LSTM(64, return_sequences=False))
        model.add(keras.layers.Dense(25))
        model.add(keras.layers.Dense(1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        return model

    def fit(self):
        dataset = self.prepare_data(self.dataframe, self.predict_column)

        training_data_len = self.training_data_len(len_dataset=len(dataset), training_data_split=self.training_data_split)
        print(f'training data {training_data_len}')

        self.scaled_data = self.min_max_scaler.fit_transform(dataset)

        x_train, y_train = self.prepare_train_data(training_data_len, self.scaled_data, self.time_steps)

        model = self.create_model(x_train)

        model.fit(x_train, y_train, batch_size=1, epochs=1, callbacks=[self.es])

        self.fit_model = model

    def predict(self):
        dataset = self.prepare_data(self.dataframe, self.predict_column)

        training_data_len = self.training_data_len(len_dataset=len(dataset),
                                                   training_data_split=self.training_data_split)
        test_data = self.scaled_data[training_data_len - self.time_steps:, :]

        x_test, y_test = self.prepare_test_data(training_data_len, test_data, dataset, self.time_steps)

        # Get the models predicted price values
        predictions = self.fit_model.predict(x_test)
        predictions = self.min_max_scaler.inverse_transform(predictions)

        rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions)

        self.predict_result['predictions'] = list(predictions)
        self.predict_result['rmse'] = rmse
        self.predict_result['r2'] = r2
        self.predict_result['mae'] = mae
        self.predict_result['mape'] = mape
