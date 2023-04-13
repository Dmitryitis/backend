from drf_yasg.utils import swagger_auto_schema
from rest_framework import viewsets, mixins, status
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from api.models import Project
from api.serializers.project.project_table import ProjectTable_ProjectTableSerializer, \
    ProjectTable_ProjectTableSmaInfoSerializer, ProjectTable_ProjectTableRollingInfoSerializer
from api.services.data_analyze.data_analyzer import DataAnalyzerOHLC


class ProjectTable_ProjectTableViewSet(viewsets.GenericViewSet,
                                       mixins.RetrieveModelMixin, ):
    permission_classes = [IsAuthenticated]

    lookup_url_kwarg = "project_id"

    def get_serializer_class(self):
        if self.action in ('retrieve',):
            return ProjectTable_ProjectTableSerializer
        elif self.action in ('get_sma',):
            return ProjectTable_ProjectTableSmaInfoSerializer
        elif self.action in ('get_rolling',):
            return ProjectTable_ProjectTableRollingInfoSerializer

    def get_queryset(self):
        if self.action in ('retrieve',):
            project_id = self.kwargs.get(self.lookup_url_kwarg)
            return Project.not_deleted.filter(id=project_id, author=self.request.user).select_related(
                "project_symbol").select_related("project_file_data")

    def retrieve(self, request, *args, **kwargs):
        project = Project.not_deleted.filter(id=self.kwargs.get(self.lookup_url_kwarg),
                                             author=self.request.user).select_related(
            "project_symbol").select_related("project_file_data")

        data_analyze = DataAnalyzerOHLC(project[0].project_file_data.file)

        data_analyze.set_index_datetime()

        print(data_analyze.get_log_profitability())

        data_analyze_dict = {
            'stat_info': {
                'rows': data_analyze.get_shape()[0],
                'columns': data_analyze.get_shape()[1],
                'columns_name': data_analyze.get_columns()
            },
            'data': data_analyze.get_head_data(),
            'describe': data_analyze.get_describe(),
            'correlation': data_analyze.get_correlation(),
        }

        serializer = self.get_serializer(data_analyze_dict)

        return Response(
            data=serializer.data,
            status=status.HTTP_200_OK,
        )

    @swagger_auto_schema(
        method="get",
        responses={200: ProjectTable_ProjectTableSmaInfoSerializer()},
        operation_summary="Get sma data.",
    )
    @action(methods=["GET"], detail=False)
    def get_sma(self, request, *args, **kwargs):
        project = Project.not_deleted.filter(id=self.kwargs.get(self.lookup_url_kwarg),
                                             author=self.request.user).select_related(
            "project_symbol").select_related("project_file_data")

        data_analyze = DataAnalyzerOHLC(project[0].project_file_data.file)

        data_analyze.set_index_datetime()

        data_analyze_dict = {
            'sma_data': data_analyze.get_sma(),
        }

        serializer = self.get_serializer(data_analyze_dict)

        return Response(
            data=serializer.data,
            status=status.HTTP_200_OK,
        )

    @swagger_auto_schema(
        method="get",
        responses={200: ProjectTable_ProjectTableRollingInfoSerializer()},
        operation_summary="Get rolling data.",
    )
    @action(methods=["GET"], detail=False)
    def get_rolling(self, request, *args, **kwargs):
        project = Project.not_deleted.filter(id=self.kwargs.get(self.lookup_url_kwarg),
                                             author=self.request.user).select_related(
            "project_symbol").select_related("project_file_data")

        data_analyze = DataAnalyzerOHLC(project[0].project_file_data.file)

        data_analyze.set_index_datetime()

        data_analyze_dict = {
            'rolling_data': data_analyze.get_rolling_statistics()
        }

        serializer = self.get_serializer(data_analyze_dict)

        return Response(
            data=serializer.data,
            status=status.HTTP_200_OK,
        )
