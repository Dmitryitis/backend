from rest_framework import serializers

from api.models import YahooSymbol


class YahooSymbols_YahooSymbolsReadSerializer(serializers.ModelSerializer):
    class Meta:
        model = YahooSymbol
        fields = (
            "symbol",
            "name"
        )