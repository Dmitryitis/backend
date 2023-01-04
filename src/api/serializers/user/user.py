from rest_framework import serializers

from api.models import User
from api.serializers.author.author import (
    Author_AuthorInfoReadSerializer,  # skip role check
)
from api.serializers.author.author import (
    Author_AuthorWriteSerializer,  # skip role check
)

class User_UserArticleReadSerializer(serializers.ModelSerializer):
    author_info = Author_AuthorInfoReadSerializer(  # skip role check
        read_only=True
    )

    class Meta:
        model = User
        fields = [
            "id",
            "author_info",
        ]
        read_only_fields = fields
        prefetch_related = (
            "author_info",
        )


class User_UserReadSerializer(serializers.ModelSerializer):
    """Serializer for getting user"""

    author_info = Author_AuthorInfoReadSerializer(  # skip role check
        read_only=True
    )

    class Meta:
        model = User
        fields = [
            "id",
            "email",
            "is_activated",
            "is_staff",
            "author_info",
            "created_at",
        ]
        read_only_fields = fields
        prefetch_related = (
            "author_info",
        )


class User_UserUpdateSerializer(serializers.ModelSerializer):
    """Serializer for update user"""

    author_info = Author_AuthorWriteSerializer(required=False)  # skip role check

    class Meta:
        model = User
        fields = [
            "email",
            "author_info",
        ]


class User_PhoneNumberUpdateSerializer(serializers.ModelSerializer):
    confirmation_code = serializers.CharField(read_only=True)

    class Meta:
        model = User
        fields = ["phone_number", "confirmation_code"]


class User_EmailUpdateSerializer(serializers.ModelSerializer):
    confirmation_code = serializers.CharField(read_only=True)

    class Meta:
        model = User
        fields = ["email", "confirmation_code"]
