import os
from datetime import datetime

from django.db import models

from api.models.abstract import BaseModel


def get_project_file_data_path(
        instance, filename: str
) -> str:  # instance: Project file data
    _, ext = os.path.splitext(filename)
    new_filename = f"{instance.pk}-{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}"
    if ext:
        new_filename += ext
    return os.path.join(
        f"project/files/data/{instance.created_at.strftime('%Y/%m/%d')}/",
        new_filename,
    )


class ProjectFileData(BaseModel):
    name = models.CharField(max_length=128, null=True, blank=True)
    size = models.CharField(max_length=64, null=True, blank=True)

    file = models.FileField(upload_to=get_project_file_data_path, null=True, blank=True)

    class Meta:
        verbose_name_plural = "ProjectFileData"
