import os
from datetime import datetime

from vkr.celery import app
from vkr import settings


def create_task(project, predict_column, project_model_id):
    fit_project.delay(project, predict_column, project_model_id)


def get_project_file_model_path(
        instance, project_id, filename: str
) -> dict[str, str]:  # instance: Project file data
    _, ext = os.path.splitext(filename)
    new_filename = f"{instance}-{project_id}-{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}"
    if ext:
        new_filename += ext
    return {"path": os.path.join(settings.MODELS, new_filename), "relative_path": new_filename }


@app.task(queue=settings.QUEUE_DEFAULT, serializer='json')
def fit_project(project_id, predict_column, project_model_id):
    from api.models import Project
    from api.services.neural_network.neural_network_lstm import NeuralNetworkLstm
    from api.services.neural_network.neural_network import NeuralNetworkFast
    from django.db import transaction
    from api.models import ProjectModel
    from api.enums import ProjectStatus
    from api.models import ProjectResult

    with transaction.atomic():

        project = Project.objects.filter(id=project_id).first()
        project_model = ProjectModel.objects.filter(id=project_model_id).first()
        neural_network = NeuralNetworkFast(project.project_file_data.file, predict_column)

        neural_network.fit()
        neural_network.predict()
        # path_dict = get_project_file_model_path(f'model',project.id, 'model')
        # path_to_save = f"{path_dict['path']}.h5"
        # neural_network.fit_model.save(path_to_save)
        #
        # project_model.save_model_url = f"{settings.BASE_URL}/media/{path_dict['relative_path']}.h5"
        # project.project_status = ProjectStatus.ready
        #
        # project_model.save()
        # project.save()
        #
        # data_result = {
        #     "predictions": neural_network.predict_result['predictions'],
        #     "rmse": neural_network.predict_result['rmse'],
        #     "r2": neural_network.predict_result['r2'],
        #     "mae": neural_network.predict_result['mae'],
        #     "mape": neural_network.predict_result['mape'],
        #     "project": project,
        #     "project_model": project_model
        # }
        #
        # ProjectResult.objects.create(**data_result)
        # print(neural_network.predict_result['r2'])
