from rest_framework import viewsets, mixins, status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from api.enums import ProjectTypeStudy, TechnicalIndicatorsEnum, ArchitectureNeuralNetworkEnum


class ProjectSettings_ProjectViewSet(viewsets.GenericViewSet,
                                     mixins.ListModelMixin, ):

    permission_classes = [IsAuthenticated]

    def list(self, request, *args, **kwargs):
        return Response(
            status=status.HTTP_200_OK,
            data={
                'mode_study': ProjectTypeStudy.values,
                'indicators': TechnicalIndicatorsEnum.values,
                'networks': ArchitectureNeuralNetworkEnum.values
            }
        )
