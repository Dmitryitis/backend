from rest_framework import viewsets, mixins, status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from api.models import Project
from api.services.data_analyze.data_analyzer import DataAnalyzerOHLC


class ProjectTable_ProjectTableViewSet(viewsets.GenericViewSet,
                                       mixins.RetrieveModelMixin, ):
    permission_classes = [IsAuthenticated]

    lookup_url_kwarg = "project_id"

    def get_queryset(self):
        if self.action in ('retrieve',):
            project_id = self.kwargs.get(self.lookup_url_kwarg)
            return Project.not_deleted.filter(id=project_id, author=self.request.user).select_related(
                "project_symbol").select_related("project_file_data")

    def retrieve(self, request, *args, **kwargs):
        project = Project.not_deleted.filter(id=self.kwargs.get(self.lookup_url_kwarg), author=self.request.user).select_related(
                "project_symbol").select_related("project_file_data")

        data_analyze = DataAnalyzerOHLC(project[0].project_file_data.file)

        print(data_analyze.get_head_data())
        print(data_analyze.get_describe())
        print(data_analyze.get_correlation())
        print(data_analyze.get_shape())

        return Response(
            status=status.HTTP_200_OK,
        )
