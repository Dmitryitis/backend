from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin
from django.contrib.postgres.indexes import HashIndex
from django.db import models
from django.utils.translation import gettext_lazy as _

from api.enums import UserRegistrationMethods
from api.models.abstract import BaseModel
from api.models.fields import LowercaseEmailField
from api.models.managers import UserManager


class User(BaseModel, AbstractBaseUser, PermissionsMixin):
    email = LowercaseEmailField(_("Email address"), unique=True, null=True, blank=True)

    is_activated = models.BooleanField(default=False)
    is_staff = models.BooleanField(default=False)

    registration_method = models.CharField(
        max_length=64,
        verbose_name=(_("Registration method: email, Google Oauth etc.")),
        choices=UserRegistrationMethods.choices,
        default=UserRegistrationMethods.email,
    )

    author_info = models.OneToOneField(
        "api.Author",
        on_delete=models.PROTECT,
        related_name="user",
        null=True,
        blank=True,
        verbose_name=_("Information about the author if it exists"),
    )

    objects = UserManager()
    USERNAME_FIELD = "email"

    class Meta:
        indexes = (
            HashIndex(fields=("email",)),
        )

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)

    def __str__(self):
        return (
            f"User #{self.pk} {self.email}"
        )
