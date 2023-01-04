import os
from io import BytesIO

from django.core.files.base import ContentFile, File
from django.core.files.storage import default_storage, Storage
from imgpy import Img
from stdimage.models import StdImageFieldFile
from stdimage.utils import render_variations

from api.services.image_convertation.image_convertation_service import convert_to_webp
from vkr.celery import app


@app.task(queue="images")
def process_photo_image(file_name, variations, replace=True):
    storage: Storage = default_storage
    if file_name.endswith(".gif"):
        render_gif_variations(file_name, variations, storage)
    else:
        file_info: File = storage.open(file_name)
        file_path = file_name

        if not file_name.endswith(".webp"):
            with file_info.open("rb") as file_descriptor:
                converted_file = convert_to_webp(file_descriptor)

            with open(converted_file.name, "rb") as converted_file_descriptor:
                path_of_webp_file = f"{os.path.splitext(file_name)[0]}.webp"
                storage.save(path_of_webp_file, converted_file_descriptor)
            file_path = path_of_webp_file

        render_variations(file_path, variations, replace=replace, storage=storage)


def render_gif_variations(file_name, variations, storage):
    for key, variation in variations.items():
        render_gif(file_name, variation, storage)


def render_gif(file_name, variation, storage):
    """
    Render `file_name` to size of `variation` and save it to `storage`
    """
    variation_name = StdImageFieldFile.get_variation_name(file_name, variation["name"])
    with storage.open(file_name) as f:
        with Img(fp=f) as img:
            if StdImageFieldFile.is_smaller(img, variation):
                img = resize_gif(variation, img)
            with BytesIO() as file_buffer:
                img.save(fp=file_buffer)
                f = ContentFile(file_buffer.getvalue())
                storage.save(variation_name, f)


def resize_gif(variation, image: Img):
    size = variation["width"], variation["height"]
    size = tuple(int(i) if i is not None else i for i in size)
    image.thumbnail(size=size)
    return image