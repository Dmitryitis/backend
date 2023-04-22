from django.contrib.postgres.fields import ArrayField
from django.db import models

from api.models.abstract import BaseModel


class ProjectResult(BaseModel):
    project = models.ForeignKey('api.Project', null=True, blank=True, on_delete=models.PROTECT)
    project_model = models.ForeignKey('api.ProjectModel', null=True, blank=True, on_delete=models.PROTECT)
    predictions = ArrayField(base_field=models.FloatField(), null=True, blank=True)
    mae = models.FloatField(null=True, blank=True)
    r2 = models.FloatField(null=True, blank=True)
    mape = models.FloatField(null=True, blank=True)
    rmse = models.FloatField(null=True, blank=True)

    class Meta:
        verbose_name_plural = "ProjectResultData"
