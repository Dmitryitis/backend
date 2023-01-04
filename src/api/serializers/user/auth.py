from rest_framework import serializers

from api.models import User
from api.serializers.fields.lower_email import LowerEmailField

class Auth_RegistrationSerializer(
    serializers.ModelSerializer,
):
    email = LowerEmailField(required=True, allow_null=False)

    class Meta:
        model = User
        fields = ["email", "password"]