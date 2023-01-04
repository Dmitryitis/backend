from rest_framework import permissions
from rest_framework.authentication import SessionAuthentication, TokenAuthentication

default_authentication_classes = (TokenAuthentication, SessionAuthentication)
default_permission_classes = (permissions.IsAuthenticated,)
