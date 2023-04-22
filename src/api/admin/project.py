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


class ProjectModelAdmin(ModelAdmin):
    list_display = ("project", 'column_predict')


class ProjectResultAdmin(ModelAdmin):
    list_display = ("id", "project", 'r2')
