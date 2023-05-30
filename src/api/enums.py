from django.db import models
from django.utils.translation import gettext_lazy as _


class UserRegistrationMethods(models.TextChoices):
    email = "email", _("Email")
    google_oauth = "google_oauth", _("Google Oauth")


class ProjectTypeData(models.TextChoices):
    file = "file", _("File")
    symbol = "symbol", _("Symbol Yahoo")


class ProjectTypeStudy(models.TextChoices):
    fast = "fast", _("Fast")
    slow = "slow", _("Slow")


class ProjectStatus(models.TextChoices):
    draft = "draft", _("draft")
    in_work = "in_work", _("in work")
    ready = "ready", _("ready")


class TechnicalIndicatorsEnum(models.TextChoices):
    bollinger_bands = 'bollinger_bands', _('bollinger_bands')
    rsi = 'rsi', _('rsi')
    fibonacci = 'fibonacci', _('fibonacci')
    ema = 'ema', _('ema')
    macd = 'macd', _('macd')

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

class ArchitectureNeuralNetworkEnum(models.TextChoices):
    lstm = 'lstm', _('lstm')
    cnnLstm = 'cnnLstm', _('cnnLstm')
    cnnBiLstm = 'cnnBiLstm', _('cnnBiLstm')

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_
