from rest_framework import serializers
from django.contrib.auth import get_user_model

User = get_user_model()

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'password')
        extra_kwargs = {'password': {'write_only': True}} # 密碼只寫入不回傳

    def create(self, validated_data):
        # 建立使用者時，必須加密密碼 (create_user 會自動處理 hash)
        user = User.objects.create_user(**validated_data)
        return user