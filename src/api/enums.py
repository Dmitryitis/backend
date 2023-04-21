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
