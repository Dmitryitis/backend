from django.db import models

from api.models.abstract import BaseModel


class YahooSymbolsCsvImport(BaseModel):
    file = models.FileField(default=None, null=True)

    class Meta:
        verbose_name = "Yahoo Symbol Csv Import"
        verbose_name_plural = "Yahoo Symbols Csv Import"

    def __str__(self):
        return self.file.name
