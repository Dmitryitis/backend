from django.db import transaction
from rest_framework import serializers

from api.enums import ProjectTypeData
from api.models import Project, ProjectFileData, YahooSymbol
from api.serializers.fields.stdimage import StdImageSerializerField
from api.serializers.yahoo_symbols.yahoo_symbols import YahooSymbols_YahooSymbolsReadSerializer
from utils.attributes import set_attribute


class ProjectFile_ProjectFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProjectFileData
        fields = ("id", "name", "size", "file")


class ProjectFile_ProjectFileWriteSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProjectFileData
        fields = ("id", "name", "size", "file")

    def create(self, validated_data):
        file_data = ProjectFileData.objects.create(**validated_data)

        return file_data


class Project_ProjectReadSerializer(serializers.ModelSerializer):
    project_cover = StdImageSerializerField(read_only=True)
    project_symbol = YahooSymbols_YahooSymbolsReadSerializer(read_only=True)
    project_file_data = ProjectFile_ProjectFileSerializer(read_only=True)

    class Meta:
        model = Project
        fields = ("id", "title", "project_cover", "project_status", "project_symbol", "project_file_data",)


class Project_ProjectCreateSerializer(serializers.ModelSerializer):
    project_cover = StdImageSerializerField(required=False)
    left = serializers.FloatField(required=False, read_only=True)
    top = serializers.FloatField(required=False, read_only=True)
    right = serializers.FloatField(required=False, read_only=True)
    bottom = serializers.FloatField(required=False, read_only=True)
    yahoo_symbol = serializers.IntegerField(required=False)
    file_data = serializers.IntegerField(required=False)

    class Meta:
        model = Project
        fields = ('title', 'yahoo_symbol', 'project_status', 'file_data',
                  "project_cover",
                  "left",
                  "top",
                  "right",
                  "bottom",)

    @staticmethod
    def _create_with_symbol(validated_data, yahoo_instance, request):
        project = Project.objects.create(**validated_data, project_symbol=yahoo_instance, author=request.user,
                                         type_data=ProjectTypeData.symbol)

        return project

    @staticmethod
    def _create_with_file_data(validated_data, file_data_instance, request):
        project = Project.objects.create(**validated_data, project_file_data=file_data_instance, author=request.user,
                                         type_data=ProjectTypeData.file)

        return project

    def create(self, validated_data):
        with transaction.atomic():
            request = self.context["request"]

            if "yahoo_symbol" in validated_data:
                yahoo_symbol_id = validated_data.pop('yahoo_symbol')
                yahoo_instance = YahooSymbol.objects.filter(id=yahoo_symbol_id).first()

                project = self._create_with_symbol(validated_data, yahoo_instance, request)

            if "file_data" in validated_data:
                file_data = validated_data.pop('file_data')
                file_data_instance = ProjectFileData.objects.filter(id=file_data).first()

                project = self._create_with_file_data(validated_data, file_data_instance, request)

        return project


class Project_ProjectUpdateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Project
        fields = ('title', 'project_status',)

    def update(self, instance, validated_data):
        with transaction.atomic():
            for attr_name, attr_value in validated_data.items():
                set_attribute(instance, attr_name, attr_value)

            instance.save()

        return instance
