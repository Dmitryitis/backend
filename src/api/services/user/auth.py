from django.contrib.auth.backends import BaseBackend
from django.contrib.auth.hashers import check_password
from rest_framework.authtoken.models import Token

from api.exceptions import (
    WrongPassword,
    WrongEmailOrPassword,
)
from api.models import User


class UserAuth(BaseBackend):
    def authenticate(self, password: str, user: User = None) -> User:
        """Authenticates user via confirmation code.
        Returns User if cannot login user returns custom APIExceptions"""
        if check_password(password, user.password):
            return user
        else:
            raise WrongPassword()


def login_user_and_return_token(email: str, password: str) -> str:
    """
    This function Login user or Create a new one if number is new.
    And returns authentication token string
    """
    try:
        user: User = User.objects.get(email=email)

        user: User = UserAuth().authenticate(
            password=password,
            user=user,
        )  # Sends user to function to reduce number of requests to database

        if not user.is_activated:  # First login for user
            user.is_activated = True
            user.save()

        token, _ = Token.objects.get_or_create(user=user)
        return token.key
    except User.DoesNotExist:  # If in any state user will be not found - login fails
        raise WrongEmailOrPassword()
