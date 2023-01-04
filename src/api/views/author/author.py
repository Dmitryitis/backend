from django.utils.translation import gettext_lazy as _
from rest_framework import mixins, viewsets, status
from rest_framework.parsers import MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from api.models import Author
from api.serializers.author.author import Author_AuthorUpdateSerializer
from api.services.helpers import crop_image


class Author_AuthorUpdateViewSet(viewsets.GenericViewSet,
                                 mixins.UpdateModelMixin, ):
    parser_classes = [MultiPartParser]

    serializer_class = Author_AuthorUpdateSerializer

    lookup_url_kwarg = "user_id"

    def get_permissions(self):
        return [IsAuthenticated()]

    def update(self, request, *args, **kwargs):
        data = self.request.data
        if all(key in data.keys() for key in ["left", "right", "top", "bottom", "avatar"]):
            data["avatar"] = crop_image(float(data["left"]),
                                        float(data["top"]),
                                        float(data["right"]),
                                        float(data["bottom"]),
                                        data["avatar"], ).open()

            partial = kwargs.pop("partial", False)
            user_id = self.kwargs.get(self.lookup_url_kwarg)
            author = Author.objects.filter(user__id=user_id).first()
            if author is None:
                return Response(_("Author not found"), status=status.HTTP_404_NOT_FOUND)
            serializer = self.get_serializer(author, data=data, partial=partial)
            serializer.is_valid(raise_exception=True)
            serializer.save()
            return Response(serializer.data)

    def partial_update(self, request, *args, **kwargs):
        kwargs["partial"] = True
        return self.update(request, *args, **kwargs)
