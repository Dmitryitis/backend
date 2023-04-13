from rest_framework import serializers


class ProjectTable_ProjectTableStatInfoSerializer(serializers.Serializer):
    rows = serializers.IntegerField()
    columns = serializers.IntegerField()
    columns_name = serializers.ListField(child=serializers.CharField(max_length=128))


class SmaInfoSerializer(serializers.Serializer):
    time = serializers.ListSerializer(child=serializers.CharField())
    data_close = serializers.ListSerializer(child=serializers.FloatField())
    sma1 = serializers.ListSerializer(child=serializers.FloatField())
    sma2 = serializers.ListSerializer(child=serializers.FloatField())
    position = serializers.ListSerializer(child=serializers.IntegerField())


class ProjectTable_ProjectTableSmaInfoSerializer(serializers.Serializer):
    sma_data = SmaInfoSerializer()


class RollingInfoSerializer(serializers.Serializer):
    time = serializers.ListSerializer(child=serializers.CharField())
    data_close = serializers.ListSerializer(child=serializers.FloatField())
    min = serializers.ListSerializer(child=serializers.FloatField())
    max = serializers.ListSerializer(child=serializers.FloatField())
    mean = serializers.ListSerializer(child=serializers.IntegerField())
    std = serializers.ListSerializer(child=serializers.IntegerField())
    ewma = serializers.ListSerializer(child=serializers.IntegerField())
    median = serializers.ListSerializer(child=serializers.IntegerField())


class ProjectTable_ProjectTableRollingInfoSerializer(serializers.Serializer):
    rolling_data = RollingInfoSerializer()


class ProjectTable_ProjectTableSerializer(serializers.Serializer):
    stat_info = ProjectTable_ProjectTableStatInfoSerializer()
    data = serializers.ListSerializer(child=serializers.ListSerializer(child=serializers.CharField(max_length=128)))
    describe = serializers.JSONField()
    correlation = serializers.JSONField()
