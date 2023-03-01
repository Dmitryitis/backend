from django.contrib.admin import ModelAdmin


class ProjectAdmin(ModelAdmin):
    list_display = (
        "id",
        "title",
    )
    search_fields = (
        "title",
    )


class ProjectFileDataAdmin(ModelAdmin):
    list_display = ("file",)