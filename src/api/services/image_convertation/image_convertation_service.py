import tempfile

from PIL import Image
from rest_framework.exceptions import ValidationError


def convert_to_webp(source_file):
    try:
        with Image.open(source_file) as img, tempfile.NamedTemporaryFile(
            suffix=".webp", delete=False
        ) as file:
            img.save(file, "webp")
    except ValueError:
        raise ValidationError()
    return file
