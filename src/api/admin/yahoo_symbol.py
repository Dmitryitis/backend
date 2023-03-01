from django.contrib.admin import ModelAdmin


class YahooSymbolAdmin(ModelAdmin):
    list_display = (
        "name",
        "symbol",
    )
    search_fields = (
        "name", "symbol"
    )
    ordering = ("name",)