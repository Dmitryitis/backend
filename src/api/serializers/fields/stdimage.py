import os

from rest_framework import serializers
from stdimage.models import StdImageFieldFile


class StdImageSerializerField(serializers.ImageField):
    """
    Get all the variations of the StdImageField
    Taken from
    https://github.com/glemmaPaul/django-stdimage-serializer/blob/master/stdimage_serializer/fields.py
    """

    def_variation = {"width": float("inf"), "height": float("inf")}

    def __init__(self, *args, variations=None, **kwargs):
        super().__init__(*args, **kwargs)
        if variations is not None:
            self.variations = {}

            self.add_variation("original", ())
            for nm, prm in list(variations.items()):
                self.add_variation(nm, prm)
        else:
            self.variations = None

    def add_variation(self, name, params):
        variation = self.def_variation.copy()
        if isinstance(params, (list, tuple)):
            variation.update(dict(zip(("width", "height"), params)))
        else:
            variation.update(params)
        self.variations[name] = variation

    def to_native(self, obj):
        return self.get_variations_urls(obj)

    def to_representation(self, obj):
        return self.get_variations_urls(obj)

    def get_variations_urls(self, obj: StdImageFieldFile):
        """
        Get all the thumbnails urls.
        """
        # model = obj.instance
        if obj.name == "" or obj.name is None:
            return {}

        return_object = {}
        field = obj.field
        # request = context.get("request")

        if hasattr(field, "variations"):
            variations = field.variations
            for key in variations.keys():
                if hasattr(obj, key):
                    field_obj = getattr(obj, key, None)
                    if field_obj and hasattr(field_obj, "url"):
                        return_object[key] = super(
                            StdImageSerializerField, self
                        ).to_representation(field_obj)
                        for key, value in return_object.items():
                            extension = os.path.splitext(value)[1]
                            # if default_storage.exists(
                            #     field_obj.name.replace(extension, ".webp")
                            # ):
                            return_object[key] = value.replace(extension, ".webp")
                            # else:
                            #     continue
        # Also include the original (if possible)
        if hasattr(obj, "url"):
            return_object["original"] = super(
                StdImageSerializerField, self
            ).to_representation(obj)

        return return_object
