from rest_framework import status
from rest_framework.exceptions import APIException
from django.utils.translation import gettext_lazy as _


class IncorrectField(APIException):
    """
    Raised when field sent by client is incorrect.
    """

    status_code = status.HTTP_400_BAD_REQUEST
    default_detail = _("Неверные данные")


class BadRequest(APIException):
    status_code = status.HTTP_400_BAD_REQUEST


class EmailAlreadyExists(APIException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_detail = _("Этот email цже существует")

class WrongEmailOrPassword(APIException):
    status_code = status.HTTP_401_UNAUTHORIZED
    default_detail = _("Неправильный email или пароль")


class WrongPassword(APIException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_detail = _("Неправильный пароль")
