"""Chatbot序列化器：用于API序列化"""

from rest_framework import serializers
from .models import ChatRoom, ChatMessage


class ChatMessageSerializer(serializers.ModelSerializer):
    """聊天消息序列化器"""
    class Meta:
        model = ChatMessage
        fields = ['id', 'role', 'content', 'timestamp']
        read_only_fields = ['id', 'timestamp']


class ChatRoomSerializer(serializers.ModelSerializer):
    """聊天室序列化器"""
    messages = ChatMessageSerializer(many=True, read_only=True)
    message_count = serializers.SerializerMethodField()

    class Meta:
        model = ChatRoom
        fields = ['id', 'title', 'created_at', 'updated_at', 'messages', 'message_count']
        read_only_fields = ['id', 'created_at', 'updated_at']

    def get_message_count(self, obj):
        """获取消息数量"""
        return obj.messages.count()


class ChatRoomListSerializer(serializers.ModelSerializer):
    """聊天室列表序列化器（不包含消息详情）"""
    message_count = serializers.SerializerMethodField()
    last_message = serializers.SerializerMethodField()

    class Meta:
        model = ChatRoom
        fields = ['id', 'title', 'created_at', 'updated_at', 'message_count', 'last_message']
        read_only_fields = ['id', 'created_at', 'updated_at']

    def get_message_count(self, obj):
        """获取消息数量"""
        return obj.messages.count()

    def get_last_message(self, obj):
        """获取最后一条消息"""
        last_msg = obj.messages.last()
        if last_msg:
            return {
                'content': last_msg.content[:50],  # 只返回前50个字符
                'role': last_msg.role,
                'timestamp': last_msg.timestamp
            }
        return None
