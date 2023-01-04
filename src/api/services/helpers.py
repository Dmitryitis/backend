from PIL import Image
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.utils.translation import gettext_lazy as _

from api.exceptions import IncorrectField
from io import BytesIO


def crop_image(left, top, right, bottom, image):
    image_to_crop = Image.open(image)
    try:
        cropped = image_to_crop.crop((left, top, right, bottom))
        thumb_io = BytesIO()
        cropped.save(thumb_io, "PNG")
        thumb_file = InMemoryUploadedFile(
            thumb_io,
            "avatar",
            "cropped_avatar.png",
            "image/png",
            thumb_io.getbuffer().nbytes,
            None,
        )
    except SystemError:
        raise IncorrectField(_("Wrong coordinates for image cropping"))
    return thumb_file
