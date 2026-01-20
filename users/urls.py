from django.urls import path
from .views import RegisterView, UserProfileView
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

urlpatterns = [
    path('register/', RegisterView.as_view(), name='register'),
    path('login/', TokenObtainPairView.as_view(), name='token_obtain_pair'), # 登入取得 Token
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'), # 刷新 Token
    path('profile/', UserProfileView.as_view(), name='profile'), # 取得/修改個人資料
]