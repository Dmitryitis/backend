from rest_framework import mixins, viewsets
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from api.permissions import default_authentication_classes
from api.serializers.user.user import User_UserReadSerializer, User_UserUpdateSerializer


class User_UserReadViewSet(viewsets.GenericViewSet, mixins.ListModelMixin):
    """Viewset for User to get information about their account"""

    authentication_classes = default_authentication_classes

    def get_serializer_class(self):
        if self.action in ("retrieve", "list"):
            return User_UserReadSerializer
        elif self.action in (
                "update",
                "partial_update",
        ):
            return User_UserUpdateSerializer

    def get_permissions(self):
        return [
            IsAuthenticated(),
        ]

    def list(self, request, *args, **kwargs):
        """Using list since we do not need any detail information only authentication"""
        serializer = self.get_serializer(self.request.user)
        return Response(serializer.data)

    def update(self, request, *args, **kwargs):
        """Update user data without pk (specified in urls)"""
        partial = kwargs.pop("partial", False)
        user = self.request.user
        serializer = self.get_serializer(user, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)

    def partial_update(self, request, *args, **kwargs):
        """Update user data without pk (specified in urls)"""
        kwargs["partial"] = True
        return self.update(request, *args, **kwargs)
