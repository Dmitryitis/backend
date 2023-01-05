from django.db import transaction
from rest_framework import serializers

from api.models import User
from api.serializers.fields.lower_email import LowerEmailField
from api.serializers.fields.stdimage import StdImageSerializerField


class Auth_RegistrationSerializer(
    serializers.ModelSerializer,
):
    email = LowerEmailField(required=True, allow_null=False)
    image = StdImageSerializerField(write_only=True)
    left = serializers.FloatField(required=False, read_only=True)
    top = serializers.FloatField(required=False, read_only=True)
    right = serializers.FloatField(required=False, read_only=True)
    bottom = serializers.FloatField(required=False, read_only=True)
    fio = serializers.CharField(max_length=32)
    nickname = serializers.CharField(max_length=128)

    class Meta:
        model = User
        fields = ["email", "password", "image", "top", "left", "right", "bottom", "fio", "nickname"]


class Auth_LoginSerializer(serializers.Serializer):
    email = LowerEmailField(required=True, allow_null=False)
    password = serializers.CharField(required=True, allow_null=False)

class LoginResponseSerializer(serializers.Serializer):
    token = serializers.CharField()