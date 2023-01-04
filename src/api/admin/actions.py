from django.contrib import admin, messages
from django.contrib.admin import ModelAdmin

from api.services.admin.image_processing_in_admin import re_render_variations


@admin.action
def convert_images(modeladmin: ModelAdmin, request, queryset):
    ids = list(queryset.values_list("id", flat=True))
    re_render_variations(ids, str(modeladmin.model.__name__))
    messages.add_message(
        request,
        messages.SUCCESS,
        f"Image convertation for {len(ids)} objects started. It will finish soon",
    )