# markets/consumers_finnhub.py
import json
import asyncio
import websockets
import finnhub
import time
from django.conf import settings
from channels.generic.websocket import AsyncWebsocketConsumer

class FinnhubConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        # 1. 取得 URL 中的 symbol (例如 AAPL)
        self.symbol = self.scope['url_route']['kwargs']['symbol'].upper()
        
        # 初始化 Finnhub Client
        self.finnhub_client = finnhub.Client(api_key=settings.FINNHUB_API_KEY)
        
        await self.accept()
        print(f"✅ [Finnhub] 前端已連線: {self.symbol}")

        # 2. 取得完整報價 (Quote) 以處理休市狀況
        # 包含: c (Current), pc (Prev Close), d (Change), dp (Percent)
        quote = await self.get_quote()
        
        if quote:
            # 儲存昨日收盤價，供後續 WebSocket 計算使用
            self.prev_close = quote.get('pc', 0)
            current_price = quote.get('c', 0)
            
            # 🔥 關鍵：連線成功當下，立刻發送一次「最後已知價格」
            # 這樣即使休市 (WebSocket 靜默)，使用者也會看到最後的收盤價，而不是空白
            await self.send(text_data=json.dumps({
                "symbol": self.symbol,
                "price": current_price,
                "change": quote.get('d', 0),
                "change_percent": quote.get('dp', 0),
                "timestamp": time.time()
            }))
        else:
            self.prev_close = 0

        # 3. 啟動 WebSocket 監聽未來的價格 (開盤後才會動)
        self.keep_running = True
        self.task = asyncio.create_task(self.finnhub_proxy())

    async def disconnect(self, close_code):
        self.keep_running = False
        if hasattr(self, 'task'):
            self.task.cancel()
        print(f"❌ [Finnhub] 前端斷線: {self.symbol}")

    async def get_quote(self):
        """
        修復：新增這個方法來取得完整報價
        因為 Finnhub 套件是同步的，所以用 run_in_executor 跑在背景
        """
        loop = asyncio.get_event_loop()
        try:
            # 呼叫 finnhub_client.quote
            quote = await loop.run_in_executor(None, self.finnhub_client.quote, self.symbol)
            return quote
        except Exception as e:
            print(f"⚠️ 無法取得 Quote: {e}")
            return None

    async def finnhub_proxy(self):
        """
        WebSocket 代理：負責轉發 Finnhub 的即時 Trade 資料
        """
        websocket_url = f"wss://ws.finnhub.io?token={settings.FINNHUB_API_KEY}"
        
        # 建立連線
        async with websockets.connect(websocket_url) as ws:
            # 訂閱
            await ws.send(json.dumps({"type": "subscribe", "symbol": self.symbol}))
            
            while self.keep_running:
                try:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    
                    # 只處理 'trade' 類型的資料
                    if data.get('type') == 'trade':
                        trades = data.get('data', [])
                        if trades:
                            # 取最新的一筆成交
                            latest = trades[-1]
                            price = latest.get('p')
                            
                            # 計算漲跌 (基於前面取得的 prev_close)
                            if self.prev_close and self.prev_close > 0:
                                change = price - self.prev_close
                                change_p = (change / self.prev_close) * 100
                            else:
                                change = 0
                                change_p = 0
                            
                            # 推送給前端
                            await self.send(text_data=json.dumps({
                                "symbol": self.symbol,
                                "price": round(price, 2),
                                "change": round(change, 2),
                                "change_percent": round(change_p, 2),
                                "timestamp": time.time()
                            }))
                except Exception as e:
                    # 連線錯誤時稍微休息再重試，避免無窮迴圈
                    print(f"WS Error: {e}")
                    await asyncio.sleep(2)