from rest_framework import generics, permissions
from .serializers import UserSerializer
from django.contrib.auth import get_user_model

User = get_user_model()

# 註冊 API (任何人都可以訪問)
class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    permission_classes = (permissions.AllowAny,)
    serializer_class = UserSerializer

# 使用者個人資料 CRUD (必須登入才能訪問)
class UserProfileView(generics.RetrieveUpdateAPIView):
    queryset = User.objects.all()
    permission_classes = (permissions.IsAuthenticated,)
    serializer_class = UserSerializer

    def get_object(self):
        # 覆寫 get_object，確保使用者只能看到/修改自己的資料
        return self.request.user