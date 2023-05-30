from django.db import transaction
from rest_framework import serializers

from api.enums import ProjectStatus, ArchitectureNeuralNetworkEnum, TechnicalIndicatorsEnum, ProjectTypeStudy
from api.models import ProjectModel, Project
from api.services.neural_network.task_network import create_task


class ProjectModel_ProjectModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProjectModel
        fields = [
            "id",
            "type_study",
            "column_predict",
            "save_model_url",
            'model',
            'indicator'
        ]


class ProjectFit_ProjectFitSerializer(serializers.ModelSerializer):
    project_id = serializers.IntegerField()
    model = serializers.ChoiceField(choices=ArchitectureNeuralNetworkEnum.choices, required=True)
    typeStudy = serializers.ChoiceField(choices=ProjectTypeStudy.choices, required=True)
    indicator = serializers.BooleanField(required=True)

    class Meta:
        model = ProjectModel
        fields = [
            "type_study",
            "column_predict",
            "project_id",
            'model',
            'indicator',
            'typeStudy'
        ]

    def create(self, validated_data):
        with transaction.atomic():
            project_id = validated_data.pop('project_id')

            type_study = validated_data.pop('type_study')
            column_predict = validated_data.pop('column_predict')
            model = validated_data.pop('model')
            indicator = validated_data.pop('indicator')
            typeStudy = validated_data.pop('typeStudy')

            project = Project.objects.filter(id=project_id).first()

            project.project_status = ProjectStatus.in_work
            project.save()

            project_model = ProjectModel.objects.create(type_study=type_study, column_predict=column_predict,
                                                        project=project, model=model, indicator=indicator)
            setting_model = {
                'indicator': indicator,
                'model': model,
                'typeStudy': typeStudy
            }
            create_task(project.id, column_predict, project_model.id, setting_model)

        return project_model
