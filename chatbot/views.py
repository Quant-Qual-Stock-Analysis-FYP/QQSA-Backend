"""Chatbot API视图：聊天和推荐问题。"""

import logging
from typing import Any

from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.generics import ListCreateAPIView, RetrieveUpdateDestroyAPIView, DestroyAPIView

from .models import ChatRoom, ChatMessage, RecommendedQuestion
from .serializers import ChatRoomSerializer, ChatRoomListSerializer, ChatMessageSerializer
from .services import generate_market_news_questions, query_deepseek

logger = logging.getLogger(__name__)


class ChatRoomListView(ListCreateAPIView):
    """聊天室列表视图：获取用户的所有聊天室，或创建新聊天室"""
    permission_classes = [IsAuthenticated]
    serializer_class = ChatRoomListSerializer

    def get_queryset(self):
        """只返回当前用户的聊天室"""
        return ChatRoom.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        """创建聊天室时自动关联当前用户"""
        serializer.save(user=self.request.user)


class ChatRoomDetailView(RetrieveUpdateDestroyAPIView):
    """聊天室详情视图：获取、更新或删除特定聊天室"""
    permission_classes = [IsAuthenticated]
    serializer_class = ChatRoomSerializer

    def get_queryset(self):
        """只返回当前用户的聊天室"""
        return ChatRoom.objects.filter(user=self.request.user)

    def destroy(self, request, *args, **kwargs):
        """删除聊天室，但确保用户至少有一个聊天室"""
        chat_room = self.get_object()
        user_chatrooms_count = ChatRoom.objects.filter(user=request.user).count()
        
        # 如果这是用户唯一的聊天室，不允许删除
        if user_chatrooms_count <= 1:
            return Response(
                {"error": "Cannot delete the last chat room. You must have at least one chat room."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        return super().destroy(request, *args, **kwargs)


class ChatView(APIView):
    """聊天视图：处理用户消息和AI回复（需要指定聊天室）"""
    permission_classes = [IsAuthenticated]

    def get(self, request, room_id):
        """取得指定聊天室的歷史對話紀錄"""
        try:
            # 确保聊天室属于当前用户
            chat_room = ChatRoom.objects.filter(id=room_id, user=request.user).first()
            if not chat_room:
                return Response(
                    {"error": "Chat room not found"},
                    status=status.HTTP_404_NOT_FOUND
                )

            messages = ChatMessage.objects.filter(chat_room=chat_room).order_by('timestamp')
            serializer = ChatMessageSerializer(messages, many=True)
            return Response(serializer.data)
        except Exception as e:
            logger.error(f"Error fetching chat history: {e}", exc_info=True)
            return Response(
                {"error": "Failed to fetch chat history"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def post(self, request, room_id):
        """發送新訊息到指定聊天室"""
        user_input = request.data.get('question')
        if not user_input:
            return Response(
                {"error": "Question is required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # 确保聊天室属于当前用户
            chat_room = ChatRoom.objects.filter(id=room_id, user=request.user).first()
            if not chat_room:
                return Response(
                    {"error": "Chat room not found"},
                    status=status.HTTP_404_NOT_FOUND
                )

            # 1. 儲存使用者的問題
            ChatMessage.objects.create(chat_room=chat_room, role='user', content=user_input)

            # 2. 準備給 AI 的 Context (抓取最近 n 則對話，避免 Token 爆掉)
            # 注意：要按時間順序傳給 LLM
            history = ChatMessage.objects.filter(chat_room=chat_room).order_by('-timestamp')[:10]
            # 因為上面是倒序抓取 (最新的在前面)，傳給 AI 前要轉正
            # 轉換 role: 'ai' -> 'assistant' (DeepSeek API 要求)
            messages_payload = [
                {
                    'role': 'assistant' if msg.role == 'ai' else msg.role,
                    'content': msg.content
                } 
                for msg in reversed(history)
            ]

            # 3. 呼叫 AI model（啟用話術轉向功能）
            # include_system_prompt=True 會啟用 Pivot System Prompt，讓 AI 能夠將非金融問題轉向到股票分析
            ai_response_text = query_deepseek(messages_payload, include_system_prompt=True)

            # 4. 儲存 AI 的回答
            ChatMessage.objects.create(chat_room=chat_room, role='ai', content=ai_response_text)

            # 5. 更新聊天室的 updated_at 时间
            chat_room.save()  # 这会自动更新 updated_at

            # 6. 回傳結果
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


class ChatMessageDeleteView(DestroyAPIView):
    """删除聊天消息视图"""
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """只返回当前用户的聊天室中的消息"""
        return ChatMessage.objects.filter(chat_room__user=self.request.user)

    def destroy(self, request, *args, **kwargs):
        """删除消息"""
        try:
            message = self.get_object()
            message.delete()
            return Response(
                {"message": "Message deleted successfully"},
                status=status.HTTP_200_OK
            )
        except Exception as e:
            logger.error(f"Error deleting message: {e}", exc_info=True)
            return Response(
                {"error": "Failed to delete message"},
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