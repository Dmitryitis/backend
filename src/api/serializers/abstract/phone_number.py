from rest_framework import serializers

from api.serializers.fields.lower_email import LowerEmailField

class Abstract_EmailSerializer(serializers.Serializer):
    email = LowerEmailField(required=True, allow_null=False)
    new_email = LowerEmailField(required=False, allow_null=False)