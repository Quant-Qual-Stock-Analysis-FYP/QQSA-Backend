"""Chatbot应用模型：聊天消息和推荐问题。"""

from django.db import models
from django.conf import settings


class ChatRoom(models.Model):
    """聊天室模型：用户可以拥有多个聊天室"""
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='chat_rooms')
    title = models.CharField(max_length=200, blank=True, help_text="聊天室标题，可选")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']  # 按更新时间倒序，最新的在前面
        verbose_name = "聊天室"
        verbose_name_plural = "聊天室"

    def __str__(self):
        title = self.title if self.title else f"Chat {self.id}"
        return f"{self.user.username} - {title}"


class ChatMessage(models.Model):
    """聊天消息模型"""
    ROLE_CHOICES = (
        ('user', 'User'),
        ('ai', 'AI'),
    )

    # 連結到聊天室 (每個聊天室有自己的消息歷史)
    chat_room = models.ForeignKey(ChatRoom, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=10, choices=ROLE_CHOICES) # 是誰說的話
    content = models.TextField() # 內容
    timestamp = models.DateTimeField(auto_now_add=True) # 時間

    class Meta:
        ordering = ['timestamp'] # 依時間排序
        verbose_name = "聊天消息"
        verbose_name_plural = "聊天消息"

    def __str__(self):
        return f"{self.chat_room} - {self.role}: {self.content[:20]}"


class RecommendedQuestion(models.Model):
    """推荐问题模型：每天更新的推荐问题"""
    question = models.CharField(max_length=500, unique=True, help_text="推荐给用户的问题")
    created_at = models.DateTimeField(auto_now_add=True, help_text="创建时间")
    updated_at = models.DateTimeField(auto_now=True, help_text="更新时间")

    class Meta:
        ordering = ['created_at']
        verbose_name = "推荐问题"
        verbose_name_plural = "推荐问题"

    def __str__(self):
        return self.question[:50]