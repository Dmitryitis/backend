from django_filters.rest_framework import FilterSet, filters

from api.enums import ProjectStatus
from api.models import Project


class BaseProjectSet(FilterSet):
    class Meta:
        model = Project
        fields = ("id",)


class ProjectFilterSet(BaseProjectSet):

    def _sort_by_status(self, queryset, name, value):
        if not value:
            return queryset

        return queryset.filter(project_status=value)

    sort_status = filters.ChoiceFilter(
        choices=ProjectStatus.choices,
        method="_sort_by_status",
        help_text="Sort by status",
    )
