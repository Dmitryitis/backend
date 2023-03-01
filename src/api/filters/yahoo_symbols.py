from django_filters.rest_framework import filters
from django_filters.rest_framework import FilterSet

from api.models import YahooSymbol


class BaseYahooSymbolsSet(FilterSet):
    class Meta:
        model = YahooSymbol
        fields = ("name", "symbol")


class YahooSymbolsFilterSet(BaseYahooSymbolsSet):

    def _sort_by_name(self, queryset, name, value):
        if not value:
            return queryset

        return queryset.order_by("name")

    def _per_page(self,queryset, name, value):
        if not value:
            return queryset

        return queryset[:value]

    sort_name = filters.BooleanFilter(
        method="_sort_by_name",
        help_text="Ordering by name yahoo symbol: [true, false]",
    )

    per_page = filters.NumberFilter(
        method="_per_page",
        help_text="Limit symbols",
    )
