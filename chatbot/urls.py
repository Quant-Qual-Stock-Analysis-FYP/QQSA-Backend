"""Chatbot URL配置"""

from django.urls import path

from .views import (
    ChatRoomListView,
    ChatRoomDetailView,
    ChatView,
    ChatMessageDeleteView,
    RecommendedQuestionsView
)

urlpatterns = [
    # 聊天室管理
    path('rooms/', ChatRoomListView.as_view(), name='chat_room_list'),
    path('rooms/<int:pk>/', ChatRoomDetailView.as_view(), name='chat_room_detail'),
    
    # 聊天消息（需要指定聊天室ID）
    path('rooms/<int:room_id>/messages/', ChatView.as_view(), name='chat_messages'),
    
    # 删除单个消息
    path('messages/<int:pk>/', ChatMessageDeleteView.as_view(), name='chat_message_delete'),
    
    # 推荐问题（保持不变）
    path('recommended-questions/', RecommendedQuestionsView.as_view(), name='recommended_questions'),
]