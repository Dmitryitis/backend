from django.db import models
from django.utils.translation import gettext_lazy as _

class UserRegistrationMethods(models.TextChoices):
    email = "email", _("Email")
    google_oauth = "google_oauth", _("Google Oauth")