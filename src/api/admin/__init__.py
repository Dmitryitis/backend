from django.contrib.admin import AdminSite

from api.admin.user import UserAdmin
from api.models import User

admin_site = AdminSite()

admin_site.register(User, UserAdmin)