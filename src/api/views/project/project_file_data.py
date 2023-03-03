from rest_framework import viewsets, mixins, status
from rest_framework.parsers import MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from api.serializers.project.project import ProjectFile_ProjectFileWriteSerializer, ProjectFile_ProjectFileSerializer


class ProjectFileData_ProjectFileDataViewSet(viewsets.GenericViewSet, mixins.CreateModelMixin):
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser]

    serializer_class = ProjectFile_ProjectFileWriteSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=self.request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        data = ProjectFile_ProjectFileSerializer(
            serializer.instance, context=self.get_serializer_context()
        ).data

        headers = self.get_success_headers(data)
        return Response(
            data,
            status=status.HTTP_201_CREATED,
            headers=headers,
        )
