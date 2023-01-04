from django.contrib.admin import ModelAdmin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from api.forms.admin.user import CustomUserCreationForm
from api.admin.actions import convert_images


class UserAdmin(BaseUserAdmin):
    """This class manages what will be visible about user in Admin panel"""

    add_form = CustomUserCreationForm
    fieldsets = ()
    add_fieldsets = (
        (
            None,
            {"classes": ("wide",), "fields": CustomUserCreationForm.Meta.fields},
        ),
    )
    list_display = (
        "id",
        "email",
        "is_superuser",
        "created_at",
        "updated_at",
    )
    list_filter = (
        "is_activated",
        "is_superuser",
        "is_staff",
    )
    search_fields = (
        "email",
    )
    ordering = ("email",)

class AuthorAdmin(ModelAdmin):
    list_display = (
        "id",
        "nickname",
        "user",
        "fio"
    )
    actions = [convert_images]
    search_fields = (
        "nickname","fio"
    )
    ordering = ("nickname",)