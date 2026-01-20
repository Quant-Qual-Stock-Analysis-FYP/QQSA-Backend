"""Chatbot URL配置"""

from django.urls import path

from .views import ChatView, RecommendedQuestionsView

urlpatterns = [
    path('ask/', ChatView.as_view(), name='chat_ask'),
    path('recommended-questions/', RecommendedQuestionsView.as_view(), name='recommended_questions'),
]