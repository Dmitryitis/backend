import os

from stdimage import StdImageField

from api.models.abstract import BaseModel
from django.db import models
from django.utils.translation import gettext_lazy as _
from datetime import datetime

from api.enums import ProjectTypeData, ProjectStatus
from api.models.managers import NotDeletedManager
from utils.image import image_processor


def get_project_cover_image_path(
        instance, filename: str
) -> str:  # instance: Project
    _, ext = os.path.splitext(filename)
    new_filename = f"{instance.pk}-{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}"
    if ext:
        new_filename += ext
    return os.path.join(
        f"project/covers/{instance.created_at.strftime('%Y/%m/%d')}/",
        new_filename,
    )


class Project(BaseModel):
    author = models.ForeignKey(
        "api.User",
        on_delete=models.PROTECT,
        related_name="project_author",
    )
    title = models.CharField(max_length=256, help_text=_("The title project"))
    type_data = models.CharField(choices=ProjectTypeData.choices, max_length=56, help_text=_("The type data project"))

    project_cover = StdImageField(
        upload_to=get_project_cover_image_path,
        verbose_name=_("AProject cover"),
        render_variations=image_processor,
        variations={
            "thumbnail": {"width": 808, "height": 632, "crop": True},
            "medium": (808, 632),
        },
        null=True,
        blank=True,
    )

    project_symbol = models.ForeignKey("api.YahooSymbol", on_delete=models.PROTECT, blank=True, null=True)
    project_file_data = models.ForeignKey("api.ProjectFileData", on_delete=models.PROTECT, blank=True, null=True)
    project_status = models.CharField(choices=ProjectStatus.choices, max_length=56,
                                      help_text=_("Project current status"))

    is_deleted = models.BooleanField(default=False)

    objects = models.Manager()
    not_deleted = NotDeletedManager()

    def delete(self, *args, **kwargs):
        self.is_deleted = True
        self.save()

    def __str__(self):
        return f"{self.title}"

    class Meta:
        verbose_name_plural = "Project"
