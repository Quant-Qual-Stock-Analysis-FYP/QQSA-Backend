"""Chatbot信号：自动创建默认聊天室"""

from django.db.models.signals import post_save
from django.dispatch import receiver
from django.conf import settings
from .models import ChatRoom


@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_default_chatroom(sender, instance, created, **kwargs):
    """当用户创建时，自动创建一个默认聊天室"""
    if created:
        # 检查用户是否已经有聊天室（防止重复创建）
        if not ChatRoom.objects.filter(user=instance).exists():
            ChatRoom.objects.create(
                user=instance,
                title="Default Chat"  # 默认标题
            )
