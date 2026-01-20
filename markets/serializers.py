# markets/serializers.py
from rest_framework import serializers
from .models import Stock, StockNews

# News Serializer
class StockNewsSerializer(serializers.ModelSerializer):
    class Meta:
        model = StockNews
        fields = ['title', 'link', 'publisher', 'icon_url', 'created_at']

# Stock Serializer
class StockSerializer(serializers.ModelSerializer):
    # 這行是關鍵：把關聯的新聞抓出來 (many=True 代表有多則新聞)
    news = StockNewsSerializer(many=True, read_only=True)

    class Meta:
        model = Stock
        fields = ['id', 'symbol', 'name', 'sector', 'details', 'last_updated', 'news']
        # 注意：上面的 fields 列表裡一定要加上 'news'