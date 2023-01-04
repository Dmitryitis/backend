import logging

from drf_yasg.utils import swagger_auto_schema
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny
from rest_framework.request import Request
from rest_framework.response import Response

from api.serializers.auth.user import Auth_LoginSerializer, LoginResponseSerializer
from api.services.user.auth import login_user_and_return_token


logger = logging.getLogger(__name__)


class Auth_LoginViewSet(
    viewsets.GenericViewSet,
):
    """ViewSet for user Authentication"""

    serializer_class = Auth_LoginSerializer

    def get_permissions(self):
        if self.action in ("email",):
            return [AllowAny()]
        raise RuntimeError()

    def login_user(self, email: str, password: str, request: Request) -> str:
        token = login_user_and_return_token(
            email=email,
            password=password,
        )
        logger.info(
            "User with phone email %s and ip address %s logged in",
            email,
            request.META["REMOTE_ADDR"],
        )
        return token

    @swagger_auto_schema(
        method="post",
        request_body=Auth_LoginSerializer(),
        responses={200: LoginResponseSerializer()},
        operation_summary="Sign in.",
    )
    @action(methods=["POST"], detail=False)
    def email(self, request, *args, **kwargs) -> Response:
        """Logins user by phone_number and confirmation code. Returns token to use in authorization"""
        serializer = self.serializer_class(data=self.request.data)
        serializer.is_valid(raise_exception=True)
        email = serializer.validated_data["email"]
        password = serializer.validated_data["password"]
        token = self.login_user(email, password, request)
        return Response(status=status.HTTP_200_OK, data={"token": token})