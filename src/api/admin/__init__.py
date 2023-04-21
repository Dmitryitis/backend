from django.contrib import admin
from django.utils.translation import gettext as _

from api.admin.project import ProjectAdmin, ProjectFileDataAdmin, ProjectModelAdmin
from api.admin.user import UserAdmin, AuthorAdmin
from api.admin.yahoo_symbol import YahooSymbolAdmin
from api.admin.yahoo_symbols_import import CsvSymbolsImportAdmin
from api.models import User, Author, YahooSymbolsCsvImport, YahooSymbol, Project, ProjectFileData, ProjectModel


class AdminSite(admin.AdminSite):
    site_title = _("Boldo Admin Panel")
    site_header = _("Boldo Admin Panel")
    index_title = _("Administration panel")
    index_template = "vkr_admin/index.html"


admin_site = AdminSite()

admin.site.register(User, UserAdmin)
admin.site.register(Author, AuthorAdmin)

admin.site.register(Project, ProjectAdmin)
admin.site.register(ProjectFileData, ProjectFileDataAdmin)
admin.site.register(ProjectModel, ProjectModelAdmin)

admin.site.register(YahooSymbolsCsvImport, CsvSymbolsImportAdmin)
admin.site.register(YahooSymbol, YahooSymbolAdmin)
