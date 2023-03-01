from django.urls import path

from api.views.project.project import Project_ProjectViewSet

urls = [
    path('project/projects/', Project_ProjectViewSet.as_view({"get": "list"}))
]