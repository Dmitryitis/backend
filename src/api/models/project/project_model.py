import os
from datetime import datetime

from django.db import models

from api.enums import ProjectTypeStudy, ArchitectureNeuralNetworkEnum
from api.models.abstract import BaseModel


def get_project_file_model_path(
        instance, filename: str
) -> str:  # instance: Project file data
    _, ext = os.path.splitext(filename)
    new_filename = f"{instance.pk}-{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}"
    if ext:
        new_filename += ext
    return os.path.join(
        f"project/models/data/{instance.created_at.strftime('%Y/%m/%d')}/",
        new_filename,
    )


class ProjectModel(BaseModel):
    project = models.ForeignKey('api.Project', on_delete=models.PROTECT, blank=True, null=True)
    column_predict = models.CharField(max_length=128, null=True, blank=True)
    type_study = models.CharField(choices=ProjectTypeStudy.choices, max_length=64)
    model = models.CharField(choices=ArchitectureNeuralNetworkEnum.choices, max_length=64, null=True, blank=True)
    indicator = models.BooleanField(default=True)
    save_model = models.FileField(upload_to=get_project_file_model_path, blank=True, null=True)
    save_model_url = models.URLField(null=True, blank=True)

    class Meta:
        verbose_name_plural = "ProjectModelData"
