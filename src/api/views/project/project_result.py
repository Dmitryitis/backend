from rest_framework import viewsets, mixins, status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from api.models import Project, ProjectResult
from api.serializers.project.project_result import ProjectResult_ProjectResultSerializer


class ProjectResult_ProjectResultViewSet(viewsets.GenericViewSet,
                                         mixins.RetrieveModelMixin):
    permission_classes = [IsAuthenticated]

    lookup_url_kwarg = "project_id"

    serializer_class = ProjectResult_ProjectResultSerializer

    def get_queryset(self):
        if self.action in ('retrieve',):
            project_id = self.kwargs.get(self.lookup_url_kwarg)
            project_instance = Project.objects.filter(id=project_id).first()
            return ProjectResult.objects.filter(project=project_instance).select_related(
                "project_model").select_related("project").first()

    def retrieve(self, request, *args, **kwargs):
        queryset = self.get_queryset()

        serializer = self.get_serializer(queryset)

        if serializer.data['project'] is not None:
            return Response(
                status=status.HTTP_200_OK,
                data=serializer.data
            )
        return Response(
            status=status.HTTP_404_NOT_FOUND,
            data={
                'status': 'not found'
            }
        )