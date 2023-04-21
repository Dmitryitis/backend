from django.conf import settings

from api.services.neural_network.neural_network import NeuralNetworkFast
from vkr.celery import app



@app.task(queue=settings.QUEUE_DEFAULT)
def fit_project(file_path, predict_column):
    neural_network = NeuralNetworkFast(file_path, predict_column)