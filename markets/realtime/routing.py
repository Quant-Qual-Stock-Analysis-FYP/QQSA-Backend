# markets/routing.py
from django.urls import re_path
from . import consumers_finnhub
from . import consumers_yahoo

websocket_urlpatterns = [
    # 🔴 路徑 1: 使用 Finnhub (預設推薦)
    re_path(r'ws/stocks/finnhub/(?P<symbol>[^/]+)/?$', consumers_finnhub.FinnhubConsumer.as_asgi()),

    # 🔴 路徑 2: 使用 Yahoo (備用)
    re_path(r'ws/stocks/yahoo/(?P<symbol>[^/]+)/?$', consumers_yahoo.YahooConsumer.as_asgi()),
    
    # (可選) 預設路徑：你可以決定 ws/stocks/{symbol} 到底要導向哪一個
    re_path(r'ws/stocks/(?P<symbol>[^/]+)/?$', consumers_yahoo.YahooConsumer.as_asgi()),
]