from tensorflow import keras

from api.enums import ArchitectureNeuralNetworkEnum


class ArchitectureNeuralNetworks:
    def __init__(self, search_params, columns_training, model):
        self.search_params = search_params
        self.columns_training = columns_training
        self.model = model

    @staticmethod
    def create_model_cnn_lstm(search_params,columns_training):
        params = search_params

        model = keras.Sequential()
        model.add(keras.layers.Conv1D(64, kernel_size=3, activation='relu',
                                      input_shape=(params['time_steps'], columns_training)))
        model.add(keras.layers.MaxPool1D(2))
        model.add(keras.layers.LSTM(params["lstm1_nodes"],
                                    dropout=0.0,
                                    recurrent_dropout=0.0,
                                    stateful=False,
                                    recurrent_regularizer=keras.regularizers.L1(0.01),
                                    return_sequences=True,
                                    input_shape=(params["time_steps"], columns_training)))
        model.add(keras.layers.LSTM(params["lstm2_nodes"], dropout=params["lstm2_dropout"], return_sequences=False))
        model.add(keras.layers.Dense(1))

        lr = params["lr"]

        if params["optimizer"] == 'rms':
            optimizer = keras.optimizers.RMSprop(learning_rate=lr)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=lr)

        # Compile the model
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    @staticmethod
    def create_model_lstm(search_params,columns_training):
        params = search_params

        model = keras.Sequential()
        model.add(keras.layers.LSTM(params["lstm1_nodes"],
                                    dropout=0.0,
                                    recurrent_dropout=0.0,
                                    stateful=False,
                                    recurrent_regularizer=keras.regularizers.L1(0.01),
                                    return_sequences=True,
                                    input_shape=(params["time_steps"], columns_training)))
        model.add(keras.layers.LSTM(params["lstm2_nodes"], dropout=params["lstm2_dropout"], return_sequences=False))
        model.add(keras.layers.Dense(1))

        lr = params["lr"]

        if params["optimizer"] == 'rms':
            optimizer = keras.optimizers.RMSprop(learning_rate=lr)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=lr)

        # Compile the model
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    @staticmethod
    def create_model_bi_lstm(search_params,columns_training):
        # Build the LSTM model

        params = search_params

        model = keras.Sequential()

        forward_layer = keras.layers.LSTM(params["lstm1_nodes"],
                                          dropout=0.0,
                                          recurrent_dropout=0.0,
                                          stateful=False,
                                          recurrent_regularizer=keras.regularizers.L1(0.01),
                                          return_sequences=True,
                                          kernel_initializer='random_uniform')
        backward_layer = keras.layers.LSTM(params["lstm1_nodes"],
                                           dropout=0.0,
                                           recurrent_dropout=0.0,
                                           stateful=False,
                                           recurrent_regularizer=keras.regularizers.L1(0.01),
                                           return_sequences=True,
                                           kernel_initializer='random_uniform',
                                           go_backwards=True)

        model.add(keras.layers.Bidirectional(forward_layer, input_shape=(params['time_steps'], columns_training),
                                             backward_layer=backward_layer, ))
        model.add(keras.layers.LSTM(params["lstm2_nodes"], dropout=params["lstm2_dropout"], return_sequences=False))
        model.add(keras.layers.Dense(1))

        lr = params["lr"]

        if params["optimizer"] == 'rms':
            optimizer = keras.optimizers.RMSprop(learning_rate=lr)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=lr)

        # Compile the model
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        return model

    def choose_model(self):
        if ArchitectureNeuralNetworkEnum.has_value(self.model):
            params = self.search_params
            columns = self.columns_training

            if self.model == ArchitectureNeuralNetworkEnum.lstm.value:
                return self.create_model_lstm(params, columns)

            if self.model == ArchitectureNeuralNetworkEnum.cnnLstm.value:
                return self.create_model_cnn_lstm(params, columns)

            if self.model == ArchitectureNeuralNetworkEnum.cnnBiLstm.value:
                return self.create_model_bi_lstm(params, columns)


