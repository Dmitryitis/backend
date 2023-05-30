from api.enums import ArchitectureNeuralNetworkEnum


class NeuralNetworkParams:
    def __init__(self, model):
        self.model = model

    @staticmethod
    def get_lstm_search_params():
        search_params = {'time_steps': 30, 'lstm1_nodes': 128, 'lstm1_dropout': 0.2, 'lstm2_dropout': 0.0,
                         'lstm2_nodes': 64, 'lr': 0.01, 'batch_size': 20, 'epochs': 30, 'optimizer': 'adam'}
        return search_params

    @staticmethod
    def get_cnn_lstm_search_params():
        search_params = {'time_steps': 60, 'lstm1_nodes': 128, 'lstm1_dropout': 0.1, 'lstm2_dropout': 0.0,
                         'lstm2_nodes': 64, 'lr': 0.01, 'batch_size': 30, 'epochs': 30, 'optimizer': 'adam'}

        return search_params

    def choose_params(self):
        if ArchitectureNeuralNetworkEnum.has_value(self.model):

            if self.model == ArchitectureNeuralNetworkEnum.lstm.value:
                return self.get_lstm_search_params()

            if self.model == ArchitectureNeuralNetworkEnum.cnnLstm.value:
                return self.get_cnn_lstm_search_params()

            if self.model == ArchitectureNeuralNetworkEnum.cnnBiLstm.value:
                return self.get_cnn_lstm_search_params()
