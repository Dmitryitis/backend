from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import mixins, viewsets
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from api.filters.project import ProjectFilterSet
from api.models import Project
from api.serializers.project.project import Project_ProjectReadSerializer


class Project_ProjectViewSet(viewsets.GenericViewSet,
                             mixins.ListModelMixin,):

    permission_classes = [IsAuthenticated]

    filter_backends = (
        DjangoFilterBackend,
    )

    filterset_class = ProjectFilterSet

    def get_serializer_class(self):
        if self.action in ("list",):
            return Project_ProjectReadSerializer
        return Project_ProjectReadSerializer

    def get_queryset(self):
        if self.action in ('list',):
            return Project.objects.filter(author=self.request.user).select_related("project_symbol").select_related("project_file_data")

    def list(self, request, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset())
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)