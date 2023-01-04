import logging

from rest_framework import status, viewsets
from rest_framework.authtoken.models import Token
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from api.serializers.abstract.phone_number import Abstract_EmailSerializer

logger = logging.getLogger(__name__)


class Auth_AuthHelpersViewSet(
    viewsets.GenericViewSet,
):
    """ViewSet for help function to authorize of authenticate. Like Get confirmation code and logout"""

    def get_permissions(self):
        if self.action in ("logout",):
            return [IsAuthenticated()]
        raise RuntimeError()

    def get_serializer_class(self):
        if self.action in ("logout",):
            return Abstract_EmailSerializer

    @action(methods=["GET"], detail=False)
    def logout(self, request, *args, **kwargs) -> Response:
        """Deleting token of user to logout user"""
        Token.objects.get(user=self.request.user).delete()
        logger.info(
            "Logout from User with phone number %s and email %s from ip address %s.",
            self.request.user.email,
            request.META["REMOTE_ADDR"],
        )
        return Response(status=status.HTTP_204_NO_CONTENT)