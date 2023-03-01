from django.db.models import UniqueConstraint

from api.models.abstract import BaseModel
from django.db import models
from django.utils.translation import gettext_lazy as _


class YahooSymbol(BaseModel):
    symbol = models.CharField(max_length=126, help_text=_("Symbol company Yahoo"))
    name = models.CharField(max_length=512, help_text=_("Name company from Yahoo"))

    def __str__(self):
        return f"{self.symbol} {self.name}"

    class Meta:
        verbose_name_plural = "YahooSymbol"
        constraints = [
            UniqueConstraint(
                fields=("symbol",),
                name="unique_constraint_for_yahoo_symbol",
            )
        ]
