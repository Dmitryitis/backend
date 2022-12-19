from django.contrib.auth.forms import UserCreationForm

from api.models import User


class CustomUserCreationForm(UserCreationForm):
    class Meta:
        model = User
        fields = (
            "email",
            "password1",
            "password2",
            "is_superuser",
            "is_activated",
            "is_staff",
        )