import os
from datetime import datetime

from django.db.models import UniqueConstraint
from django.utils.translation import gettext_lazy as _
from stdimage import StdImageField
from django.db import models

from api.models.abstract import BaseModel
from utils.image import image_processor


def get_author_avatar_image_path(
    instance, filename: str
) -> str:  # instance: Author
    _, ext = os.path.splitext(filename)
    new_filename = f"{instance.pk}-{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}"
    if ext:
        new_filename += ext
    return os.path.join(
        f"author/images/{instance.created_at.strftime('%Y/%m/%d')}/",
        new_filename,
    )

class Author(BaseModel):
    # Related name "user" to api.models.user.User
    avatar = StdImageField(
        upload_to=get_author_avatar_image_path,
        verbose_name=_("Author avatar"),
        render_variations=image_processor,
        variations={
            "thumbnail": {"width": 152, "height": 152, "crop": True},
            "medium": (400, 400),
        },
        null=True,
        blank=True,
    )
    nickname = models.CharField(
        max_length=64, help_text=_("The nickname of the author")
    )
    fio = models.CharField(
        max_length=255, help_text=_("Fio of the author"), default=""
    )

    def __str__(self):
        return self.nickname

    class Meta:
        verbose_name_plural = "Author"
        constraints = [
            UniqueConstraint(
                fields=("nickname",),
                name="unique_constraint_for_author",
            )
        ]
