from django.urls import path

from api.views.author.author import Author_AuthorUpdateViewSet

urls = [
    path(
        "author/update/",
        Author_AuthorUpdateViewSet.as_view({'put': 'update'}),
        name="author-info-update"
    ),
]