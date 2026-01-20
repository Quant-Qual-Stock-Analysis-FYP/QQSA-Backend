from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import StockViewSet

router = DefaultRouter()

# router.register(r'stocks', StockViewSet) 這行代碼自動產生了以下 URL：
#
# GET /api/stocks/        (列出所有股票)
# GET /api/stocks/{id}/   (取得單一股票詳情)
# 
# 以及你在 views.py 用 @action 裝飾器定義的所有功能：
# GET /api/stocks/by_symbol/
# GET /api/stocks/chart/

router.register(r'stocks', StockViewSet)

urlpatterns = [
    path('', include(router.urls)),
]