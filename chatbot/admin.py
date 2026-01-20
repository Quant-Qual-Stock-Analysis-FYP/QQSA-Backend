"""Chatbot应用管理界面配置"""

from django.contrib import admin

from .models import ChatMessage, RecommendedQuestion


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    """聊天消息管理"""
    list_display = ('user', 'role', 'content_preview', 'timestamp')
    list_filter = ('role', 'timestamp')
    search_fields = ('user__username', 'content')
    readonly_fields = ('timestamp',)
    
    def content_preview(self, obj):
        return obj.content[:50] + "..." if len(obj.content) > 50 else obj.content
    content_preview.short_description = "内容预览"


@admin.register(RecommendedQuestion)
class RecommendedQuestionAdmin(admin.ModelAdmin):
    """推荐问题管理"""
    list_display = ('id', 'question_preview', 'created_at', 'updated_at')
    list_filter = ('created_at', 'updated_at')
    search_fields = ('question',)
    readonly_fields = ('created_at', 'updated_at')
    
    def question_preview(self, obj):
        return obj.question[:50] + "..." if len(obj.question) > 50 else obj.question
    question_preview.short_description = "问题预览"
