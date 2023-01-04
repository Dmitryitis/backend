from rest_framework import serializers

from api.serializers.fields.stdimage import StdImageSerializerField

from api.models import Author


class Author_AuthorWriteSerializer(serializers.ModelSerializer):
    """Serializer for creating influencer"""

    avatar = StdImageSerializerField(required=False)
    left = serializers.FloatField(required=False, read_only=True)
    top = serializers.FloatField(required=False, read_only=True)
    right = serializers.FloatField(required=False, read_only=True)
    bottom = serializers.FloatField(required=False, read_only=True)
    fio = serializers.CharField(required=False)
    nickname = serializers.CharField(required=False)

    class Meta:
        model = Author
        fields = [
            "avatar",
            "nickname",
            "left",
            "top",
            "right",
            "bottom",
            "fio",
        ]


class Author_AuthorUpdateSerializer(Author_AuthorWriteSerializer):
    avatar = StdImageSerializerField(required=False, allow_null=True)
    nickname = serializers.CharField(required=False)
    fio = serializers.CharField(required=False)


class Author_AuthorInfoReadSerializer(serializers.ModelSerializer):
    """Serializer for getting user influencer"""

    avatar = StdImageSerializerField(read_only=True)

    class Meta:
        model = Author
        fields = ["avatar", "nickname", "fio"]
        read_only_fields = fields
