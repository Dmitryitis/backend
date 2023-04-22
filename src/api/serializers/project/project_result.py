from rest_framework import serializers

from api.models import ProjectResult
from api.serializers.project.project_fit import ProjectModel_ProjectModelSerializer


class ProjectResult_ProjectResultSerializer(serializers.ModelSerializer):

    project_model = ProjectModel_ProjectModelSerializer()

    class Meta:
        model = ProjectResult
        fields = ("id", "project", "predictions", "mae", 'r2', 'mape', 'rmse', 'project_model')