from django.urls import path

from api.views.yahoo.symbols import YahooSymbols_YahooSymbolsViewsSet

urls = [
    path("symbols/search/", YahooSymbols_YahooSymbolsViewsSet.as_view({"get": "list"}))
]
