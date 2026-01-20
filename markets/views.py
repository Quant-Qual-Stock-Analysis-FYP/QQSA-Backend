# markets/views.py
import time
import pytz 
from datetime import datetime
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Stock
from .serializers import StockSerializer
from yahooquery import Ticker
from .services.market_time import get_market_status

class StockViewSet(viewsets.ModelViewSet):
    # 使用 prefetch_related 優化查詢效能
    queryset = Stock.objects.all().prefetch_related('news')
    serializer_class = StockSerializer

    @action(detail=False, methods=['get'])
    def by_symbol(self, request):
        symbol = request.query_params.get('symbol')
        if not symbol:
            return Response({'error': 'Please provide symbol'}, status=status.HTTP_400_BAD_REQUEST)
        
        symbol = symbol.upper()
        
        # 🔥 优化：使用 prefetch_related 优化查询
        stock = Stock.objects.filter(symbol=symbol).prefetch_related('news').first()
        if stock:
            serializer = StockSerializer(stock)
            return Response(serializer.data)
        return Response({"error": "Stock not found"}, status=status.HTTP_404_NOT_FOUND)

    @action(detail=False, methods=['get'])
    def chart(self, request):
        symbol = request.query_params.get('symbol')
        if not symbol:
            return Response({'error': 'Missing symbol'}, status=status.HTTP_400_BAD_REQUEST)
            
        symbol = symbol.upper()
        
        # 移除緩存相關代碼
        
        # 處理加密貨幣符號 (例如 BINANCE:BTCUSDT -> BTC-USD)
        # Yahoo 不認得 Finnhub 的格式
        if 'BINANCE:' in symbol:
            yahoo_symbol = symbol.replace('BINANCE:', '').replace('USDT', '-USD')
        else:
            yahoo_symbol = symbol

        try:
            ticker = Ticker(yahoo_symbol)
            
            # 🔥 關鍵策略：抓取 "5天" (5d) 而不是 1天
            # 原因：如果是週末 (週六/日)，抓 1d 會回傳空值。
            # 抓 5d 可以確保我們一定能抓到 "上週五" 的數據。
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}?range=5d&interval=1m&includePrePost=false"
            
            # 使用 yahooquery 的 session 發送請求 (偽裝成瀏覽器，較不易被擋)
            response = ticker.session.get(url)
            
            if response.status_code != 200:
                return Response({"error": "Failed to fetch data from Yahoo"}, status=response.status_code)

            data = response.json()
            result = data.get('chart', {}).get('result', [])
            
            if not result:
                return Response([], status=status.HTTP_200_OK)

            # 解析 Yahoo 回傳的 JSON 結構
            quote_data = result[0]
            timestamps = quote_data.get('timestamp', [])
            indicators = quote_data.get('indicators', {}).get('quote', [{}])[0]
            closes = indicators.get('close', [])
            
            # 設定時區
            utc_tz = pytz.UTC
            nyc_tz = pytz.timezone('America/New_York')

            chart_data = []
            valid_date_str = ""  # 用來記錄這張圖表是哪一天的

            # 🔍 過濾邏輯：只取 "最後一個交易日" 的數據
            # 如果我們不這樣做，圖表會把過去 5 天的線全部擠在一起，變得很難看
            if timestamps:
                # 1. 找出最後一筆數據的日期 (例如 2023-10-27)
                last_ts = timestamps[-1]
                last_date_nyc = datetime.fromtimestamp(last_ts, utc_tz).astimezone(nyc_tz).date()
                valid_date_str = last_date_nyc.strftime('%Y-%m-%d')

                for i, ts in enumerate(timestamps):
                    if i < len(closes) and closes[i] is not None:
                        # 轉成紐約時間
                        dt_utc = datetime.fromtimestamp(ts, utc_tz)
                        dt_nyc = dt_utc.astimezone(nyc_tz)
                        
                        # 2. 只保留跟 "最後一筆數據" 同一天的資料
                        if dt_nyc.date() == last_date_nyc:
                            # 格式化為 "HH:MM" (前端只需要時間)
                            # 如果前端需要日期區分，可以改回 "%Y-%m-%d %H:%M"
                            time_str = dt_nyc.strftime('%H:%M')
                            
                            chart_data.append({
                                'time': time_str,
                                'price': round(closes[i], 2)
                            })

            # 🔥 關鍵修改：回傳更多 Meta Data
            is_open, status_label = get_market_status()
            
            response_data = {
                "meta": {
                    "symbol": symbol,
                    "market_status": status_label,   # "Market Open" or "Closed"
                    "is_market_open": is_open,       # true/false (前端可用來顯示紅綠燈)
                    "data_date": valid_date_str,     # "2023-10-27" (前端顯示：這是哪一天的圖)
                    "range_desc": "1 Day (Intraday)",
                    "server_timestamp": time.time()
                },
                "data": chart_data
            }

            # 移除 chart 資料 caching
            
            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=500)