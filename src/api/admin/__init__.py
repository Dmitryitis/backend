from django.contrib.admin import AdminSite

from api.admin.user import UserAdmin, AuthorAdmin
from api.models import User, Author

admin_site = AdminSite()

admin_site.register(User, UserAdmin)
admin_site.register(Author, AuthorAdmin)