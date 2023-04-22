from rest_framework import viewsets, mixins, status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from api.models import ProjectModel, Project
from api.serializers.project.project_fit import ProjectModel_ProjectModelSerializer


class ProjectModel_ProjectModelViewSet(viewsets.GenericViewSet,
                                       mixins.RetrieveModelMixin):
    permission_classes = [IsAuthenticated]

    lookup_url_kwarg = "project_id"

    serializer_class = ProjectModel_ProjectModelSerializer

    def get_queryset(self):
        if self.action in ('retrieve',):
            project_id = self.kwargs.get(self.lookup_url_kwarg)
            project_instance = Project.objects.filter(id=project_id).first()
            return ProjectModel.objects.filter(project=project_instance).first()

    def retrieve(self, request, *args, **kwargs):
        queryset = self.get_queryset()

        serializer = self.get_serializer(queryset)

        if serializer.data["type_study"] is not None:

            return Response(
                status=status.HTTP_200_OK,
                data=serializer.data
            )

        return Response(
            status=status.HTTP_404_NOT_FOUND,
            data={
                "status": 'not found'
            }
        )