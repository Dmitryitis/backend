import logging

from django.utils.translation import gettext_lazy as _
from rest_framework.pagination import PageNumberPagination

from api.exceptions import IncorrectField
from vkr import settings

logger = logging.getLogger(__name__)


class StrictLimitOffsetPagination(PageNumberPagination):
    """
    Strict version of LimitOffsetPagination.
    Instead of cutting limit by max_limit raise IncorrectField exception.
    """

    page_size = 10
    max_page_size = None

    def get_page_size(self, request):
        """
        Return limit for pagination.
        Because max_limit is None, super().get_limit(request) will return
        non-cut value of limit from querystring. Due to this, we can detect requests with
        limit greater than in settings.MAX_PAGE_SIZE and raise IncorrectField.
        """

        limit = super().get_page_size(request)

        if limit > settings.MAX_PAGE_SIZE:
            logger.warning(
                "Request %s has limit parameter %s",
                request,
                limit,
            )
            raise IncorrectField(
                _("Maximum limit is %(max_page_size)s")
                % {"max_page_size": settings.MAX_PAGE_SIZE}
            )

        return limit
