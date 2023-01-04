from django.urls import path

from api.views.user.user import User_UserReadViewSet

urls = [
    # Registering User viewset like this to update without pk
    path(
        "user/info/",
        User_UserReadViewSet.as_view(
            {"get": "list", "put": "update", "patch": "partial_update"}
        ),
    ),
]