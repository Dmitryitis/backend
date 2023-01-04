from rest_framework import serializers


class LowerEmailField(serializers.EmailField):
    def to_internal_value(self, data):
        return super().to_internal_value(data).lower()

    def to_representation(self, value):
        return super().to_representation(value).lower()