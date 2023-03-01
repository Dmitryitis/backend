from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import viewsets, mixins
from rest_framework.filters import SearchFilter
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from api.filters.yahoo_symbols import YahooSymbolsFilterSet
from api.models import YahooSymbol
from api.serializers.yahoo_symbols.yahoo_symbols import YahooSymbols_YahooSymbolsReadSerializer


class YahooSymbols_YahooSymbolsViewsSet(viewsets.GenericViewSet,
                             mixins.ListModelMixin,):

    permission_classes = [AllowAny]

    LIMIT = 100

    serializer_class = YahooSymbols_YahooSymbolsReadSerializer

    queryset = YahooSymbol.objects.all()

    search_fields = ["symbol", "name"]

    filter_backends = (
        SearchFilter,
        DjangoFilterBackend,
    )

    filterset_class = YahooSymbolsFilterSet

    def list(self, request, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset())
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)