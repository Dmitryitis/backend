from rest_framework.routers import DefaultRouter

from api.views.auth.auth import Auth_AuthHelpersViewSet

# Auth API
from api.views.auth.login import Auth_LoginViewSet
from api.views.auth.register import Auth_RegisterViewSet

router = DefaultRouter()

router.register("auth", Auth_AuthHelpersViewSet, "auth-auth")
router.register("auth", Auth_RegisterViewSet, "auth-register")
router.register("auth/login", Auth_LoginViewSet, "auth-login")

urls = router.urls