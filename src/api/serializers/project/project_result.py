from rest_framework import serializers

from api.models import ProjectResult, Project
from api.serializers.project.project import ProjectFile_ProjectFileSerializer
from api.serializers.project.project_fit import ProjectModel_ProjectModelSerializer


class ProjectForResult_ProjectForResultSerializer(serializers.ModelSerializer):
    project_file_data = ProjectFile_ProjectFileSerializer(read_only=True)

    class Meta:
        model = Project
        fields = ("id", "project_file_data")


class ProjectResult_ProjectResultSerializer(serializers.ModelSerializer):
    project_model = ProjectModel_ProjectModelSerializer()
    project = ProjectForResult_ProjectForResultSerializer()

    class Meta:
        model = ProjectResult
        fields = ("id", "mae", "mae_train", 'r2', 'r2_train', 'mape', 'mape_train', 'rmse', 'rmse_train', "project", "predictions", 'project_model', 'loss', 'val_loss')
