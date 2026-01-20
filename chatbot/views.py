"""Chatbot API视图：聊天和推荐问题。"""

import logging
from typing import Any

from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import ChatMessage, RecommendedQuestion
from .services import generate_market_news_questions, query_deepseek

logger = logging.getLogger(__name__)


class ChatView(APIView):
    """聊天视图：处理用户消息和AI回复"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """取得歷史對話紀錄"""
        try:
            messages = (
                ChatMessage.objects.filter(user=request.user)
                .values('role', 'content', 'timestamp')
                .order_by('timestamp')
            )
            return Response(list(messages))
        except Exception as e:
            logger.error(f"Error fetching chat history: {e}", exc_info=True)
            return Response(
                {"error": "Failed to fetch chat history"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def post(self, request):
        """發送新訊息"""
        user_input = request.data.get('question')
        if not user_input:
            return Response(
                {"error": "Question is required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            user = request.user

            # 1. 儲存使用者的問題
            ChatMessage.objects.create(user=user, role='user', content=user_input)

            # 2. 準備給 AI 的 Context (抓取最近 n 則對話，避免 Token 爆掉)
            # 注意：要按時間順序傳給 LLM
            history = ChatMessage.objects.filter(user=user).order_by('-timestamp')[:1]
            # 因為上面是倒序抓取 (最新的在前面)，傳給 AI 前要轉正
            messages_payload = [
                {'role': msg.role, 'content': msg.content} 
                for msg in reversed(history)
            ]

            # 3. 呼叫 AI model（啟用話術轉向功能）
            # include_system_prompt=True 會啟用 Pivot System Prompt，讓 AI 能夠將非金融問題轉向到股票分析
            ai_response_text = query_deepseek(messages_payload, include_system_prompt=True)

            # 4. 儲存 AI 的回答
            ChatMessage.objects.create(user=user, role='ai', content=ai_response_text)

            # 5. 回傳結果
            return Response({
                "question": user_input,
                "answer": ai_response_text
            })
        except Exception as e:
            logger.error(f"Error processing chat message: {e}", exc_info=True)
            return Response(
                {"error": "Failed to process message"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class RecommendedQuestionsView(APIView):
    """推荐问题视图：获取每日更新的推荐问题"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """获取推荐问题列表"""
        try:
            questions = (
                RecommendedQuestion.objects.all()
                .values('id', 'question')
                .order_by('created_at')
            )
            return Response({
                "questions": list(questions)
            })
        except Exception as e:
            logger.error(f"Error fetching recommended questions: {e}", exc_info=True)
            return Response(
                {"error": "Failed to fetch recommended questions"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )