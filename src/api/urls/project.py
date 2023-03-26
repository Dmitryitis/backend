from django.urls import path
from rest_framework.routers import DefaultRouter

from api.views.project.project import Project_ProjectViewSet
from api.views.project.project_file_data import ProjectFileData_ProjectFileDataViewSet
from api.views.project.project_table import ProjectTable_ProjectTableViewSet

router = DefaultRouter()

router.register('project', Project_ProjectViewSet, 'project-project')

urls = [
    path('project/file/', ProjectFileData_ProjectFileDataViewSet.as_view({'post': 'create'})),
    path('project/<int:project_id>/table/', ProjectTable_ProjectTableViewSet.as_view({'get': 'retrieve'}))
]

urls.extend([*router.urls])
