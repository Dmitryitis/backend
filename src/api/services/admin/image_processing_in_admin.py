from django.core.files.storage import default_storage

from api.models import (
    Author,
)
from utils.image import image_processor

CLASSES = [
    Author,
]


def re_render_variations(ids, model_name):
    for _class in CLASSES:
        if _class.__name__ == model_name:
            model_class = _class
    data = model_class.objects.filter(id__in=ids)

    if model_class.__name__ == Author.__name__:
        for item in data:
            if item.avatar:
                try:
                    variations = item.avatar.field.variations
                    image_processor(item.avatar.name, variations, default_storage)
                except FileNotFoundError:
                    continue
