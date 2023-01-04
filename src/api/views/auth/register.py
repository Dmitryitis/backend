import logging
from typing import Tuple

from django.db import IntegrityError, transaction
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from api.enums import UserRegistrationMethods
from api.exceptions import BadRequest, EmailAlreadyExists
from api.models import User
from api.serializers.auth.user import Auth_RegistrationSerializer

logger = logging.getLogger(__name__)


class Auth_RegisterViewSet(viewsets.GenericViewSet):
    def get_serializer_class(self):
        return Auth_RegistrationSerializer

    def get_permissions(self):
        return [AllowAny()]

    def _validate_serializer_data_to_uniqueness(
            self,
            serializer,
    ) -> Tuple[str, str]:
        serializer.is_valid(raise_exception=True)
        email = serializer.validated_data["email"]
        if User.objects.filter(email=email).first():
            raise EmailAlreadyExists()

        return email

    def _create_user(self, registration_fields: dict):
        with transaction.atomic():
            try:
                user = User.objects.create_user(**registration_fields)

            except IntegrityError:
                raise BadRequest("Email already taken")
            logger.info(
                f"Registered new user: {user.email}"
            )
            return user

    @action(methods=["POST"], detail=False)
    @transaction.atomic
    def register(self, request, *args, **kwargs) -> Response:
        """Registries buyer via serializer data"""
        serializer = self.get_serializer(data=self.request.data)
        email = self._validate_serializer_data_to_uniqueness(serializer)

        registration_fields = {
            "email": email,
            "password": serializer.validated_data.get("password"),
            "registration_method": UserRegistrationMethods.email,
        }
        user = self._create_user(registration_fields)
        return Response(
            {"detail": "success"},
            status=status.HTTP_201_CREATED,
        )