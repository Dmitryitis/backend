from django.contrib import admin
from django.utils.translation import gettext as _


from api.admin.user import UserAdmin, AuthorAdmin
from api.models import User, Author

class AdminSite(admin.AdminSite):
    site_title = _("Boldo Admin Panel")
    site_header = _("Boldo Admin Panel")
    index_title = _("Administration panel")
    index_template = "vkr_admin/index.html"

admin_site = AdminSite()

admin.site.register(User, UserAdmin)
admin.site.register(Author, AuthorAdmin)