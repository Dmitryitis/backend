from django.urls import path
from rest_framework.routers import DefaultRouter

from api.views.project.project import Project_ProjectViewSet
from api.views.project.project_file_data import ProjectFileData_ProjectFileDataViewSet

router = DefaultRouter()

router.register('project', Project_ProjectViewSet, 'project-project')

urls = [
    path('project/file/', ProjectFileData_ProjectFileDataViewSet.as_view({'post': 'create'}))
]

urls.extend([*router.urls])