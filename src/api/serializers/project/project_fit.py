from django.db import transaction
from rest_framework import serializers

from api.enums import ProjectStatus
from api.models import ProjectModel, Project


class ProjectModel_ProjectModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProjectModel
        fields = [
            "id",
            "type_study",
            "column_predict",
        ]


class ProjectFit_ProjectFitSerializer(serializers.ModelSerializer):
    project_id = serializers.IntegerField()

    class Meta:
        model = ProjectModel
        fields = [
            "type_study",
            "column_predict",
            "project_id"
        ]

    def create(self, validated_data):
        with transaction.atomic():
            project_id = validated_data.pop('project_id')

            type_study = validated_data.pop('type_study')
            column_predict = validated_data.pop('column_predict')

            project = Project.objects.filter(id=project_id).first()

            project.project_status = ProjectStatus.in_work
            project.save()

            project_model = ProjectModel.objects.create(type_study=type_study, column_predict=column_predict,
                                                        project=project)

        return project_model
