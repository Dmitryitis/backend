from django.contrib.auth.base_user import BaseUserManager
from django.db.models import Manager
from django.utils.timezone import now


class NotYesterdayAndDeleteManager(Manager):
    def get_queryset(self):
        return (
            super()
                .get_queryset()
                .filter(
                task_date__gte=now().date(),
                is_deleted=False
            )
        )


class UserManager(BaseUserManager):
    def _create_user(self, email: str, password=None, commit=True, **extra_fields):
        """
        Create and save a user with the given email and name.
        """
        user = self.model(email=email, **extra_fields)
        if password is not None:
            user.set_password(password)
        if commit:
            user.save(using=self._db)
        return user

    def create_user(self, email: str, **extra_fields):
        """This function creates regular user client (buyer)."""
        extra_fields.setdefault("is_staff", False)
        extra_fields.setdefault("is_superuser", False)
        return self._create_user(email, **extra_fields)

    def create_superuser(self, email: str, password, **extra_fields):
        """This function helps to create superuser with needed privileges"""
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)
        extra_fields.setdefault("is_activated", True)

        if extra_fields.get("is_staff") is not True:
            raise ValueError("Superuser must have is_staff=True.")
        if extra_fields.get("is_superuser") is not True:
            raise ValueError("Superuser must have is_superuser=True.")

        return self._create_user(email, password, **extra_fields)


class NotDeletedManager(Manager):
    def get_queryset(self):
        return (
            super()
                .get_queryset()
                .filter(is_deleted=False)
        )
