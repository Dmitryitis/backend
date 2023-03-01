from rest_framework import serializers

from api.models import Project, ProjectFileData
from api.serializers.fields.stdimage import StdImageSerializerField
from api.serializers.yahoo_symbols.yahoo_symbols import YahooSymbols_YahooSymbolsReadSerializer


class ProjectFile_ProjectFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProjectFileData
        fields = ("name", "size", "file")


class Project_ProjectReadSerializer(serializers.ModelSerializer):
    project_cover = StdImageSerializerField(read_only=True)
    project_symbol = YahooSymbols_YahooSymbolsReadSerializer(read_only=True)
    project_file_data = ProjectFile_ProjectFileSerializer(read_only=True)

    class Meta:
        model = Project
        fields = ("title", "project_cover", "project_status", "project_symbol", "project_file_data")
