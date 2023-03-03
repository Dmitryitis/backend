from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import mixins, viewsets, status
from rest_framework.parsers import MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from api.filters.project import ProjectFilterSet
from api.models import Project
from api.serializers.project.project import Project_ProjectReadSerializer, Project_ProjectCreateSerializer, \
    Project_ProjectUpdateSerializer
from api.services.helpers import crop_image


class Project_ProjectViewSet(viewsets.GenericViewSet,
                             mixins.ListModelMixin,
                             mixins.RetrieveModelMixin,
                             mixins.UpdateModelMixin,
                             mixins.CreateModelMixin,
                             mixins.DestroyModelMixin):

    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser]

    filter_backends = (
        DjangoFilterBackend,
    )

    filterset_class = ProjectFilterSet

    lookup_url_kwarg = "project_id"

    def get_serializer_class(self):
        if self.action in ("list", "retrieve"):
            return Project_ProjectReadSerializer
        elif self.action in ("create",):
            return Project_ProjectCreateSerializer
        elif self.action in ("update",):
            return Project_ProjectUpdateSerializer
        return Project_ProjectReadSerializer

    def get_queryset(self):
        if self.action in ('list',):
            return Project.not_deleted.filter(author=self.request.user).select_related("project_symbol").select_related("project_file_data")
        elif self.action in ('retrieve',):
            project_id = self.kwargs.get(self.lookup_url_kwarg)
            return Project.not_deleted.filter(id=project_id, author=self.request.user).select_related("project_symbol").select_related("project_file_data")

    def list(self, request, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset())
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=self.request.data)
        data = self.request.data
        if all(
                key in data.keys() for key in ["left", "right", "top", "bottom", "project_cover"]
        ):
            data["project_cover"] = crop_image(
                float(data["left"]),
                float(data["top"]),
                float(data["right"]),
                float(data["bottom"]),
                data["project_cover"],
            ).open()
        serializer.is_valid(raise_exception=True)

        self.perform_create(serializer)

        headers = self.get_success_headers(data)
        return Response(
            status=status.HTTP_201_CREATED,
            headers=headers,
        )