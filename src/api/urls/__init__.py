from django.urls import re_path, path

from api.swagger import schema_view
from .auth import urls as auth_urls
from .author import urls as author_urls
from .user import urls as user_urls

urlpatterns = [
    path(
        "swagger/",
        schema_view.with_ui("swagger", cache_timeout=0),
        name="schema-swagger-ui",
    ),
    re_path(
        r"^swagger(?P<format>\.json|\.yaml)$",
        schema_view.without_ui(cache_timeout=0),
        name="schema-json",
    ),
    path(
        "redoc/",
        schema_view.with_ui("redoc", cache_timeout=0),
        name="schema-redoc",
    ),
]

urlpatterns.extend(auth_urls)
urlpatterns.extend(author_urls)
urlpatterns.extend(user_urls)
