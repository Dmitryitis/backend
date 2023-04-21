from rest_framework import viewsets, mixins, status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from api.models import ProjectModel
from api.serializers.project.project_fit import ProjectFit_ProjectFitSerializer, ProjectModel_ProjectModelSerializer


class ProjectFit_ProjectViewSet(viewsets.GenericViewSet,
                                mixins.CreateModelMixin, ):
    serializer_class = ProjectFit_ProjectFitSerializer

    permission_classes = [IsAuthenticated]

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=self.request.data)

        serializer.is_valid(raise_exception=True)

        is_have_project_model = ProjectModel.objects.filter(project=self.request.data['project_id'])

        data = {
            "status": 'модель уже существует'
        }
        if len(is_have_project_model) == 0:
            self.perform_create(serializer)

            data = ProjectModel_ProjectModelSerializer(
                serializer.instance, context=self.get_serializer_context()
            ).data

            return Response(
                status=status.HTTP_201_CREATED,
                data=data
            )

        return Response(
            status=status.HTTP_200_OK,
            data=data
        )