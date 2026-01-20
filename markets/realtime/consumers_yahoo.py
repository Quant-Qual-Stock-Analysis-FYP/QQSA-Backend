# markets/consumers_yahoo.py
import json
import asyncio
import time
from channels.generic.websocket import AsyncWebsocketConsumer
from yahooquery import Ticker
from ..services.market_time import get_market_status  # 匯入剛剛寫的工具

class YahooConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.symbol = self.scope['url_route']['kwargs']['symbol'].upper()
        
        if 'BINANCE:' in self.symbol:
            self.yahoo_symbol = self.symbol.replace('BINANCE:', '').replace('USDT', '-USD')
        else:
            self.yahoo_symbol = self.symbol

        # 1. 初始化一個變數，用來記錄上一次發送的數據
        self.last_sent_values = None 

        await self.accept()
        print(f"✅ [Yahoo] 前端已連線: {self.symbol}")
        
        # 連線後立刻發送一次 (不管有沒有重複，第一次一定要發)
        await self.send_stock_data(force_send=True)

        self.keep_running = True
        self.task = asyncio.create_task(self.yahoo_poll())

    async def disconnect(self, close_code):
        self.keep_running = False
        if hasattr(self, 'task'):
            self.task.cancel()
        print(f"❌ [Yahoo] 前端斷線: {self.symbol}")

    async def get_yahoo_data(self):
        loop = asyncio.get_event_loop()
        try:
            ticker = Ticker(self.yahoo_symbol)
            # 使用 lambda 獲取 price 屬性
            data = await loop.run_in_executor(None, lambda: ticker.price)
            return data
        except Exception as e:
            print(f"⚠️ [Yahoo] API Error: {e}")
            return None

    async def send_stock_data(self, force_send=False):
        price_data = await self.get_yahoo_data()
        
        if price_data and isinstance(price_data, dict) and self.yahoo_symbol in price_data:
            data = price_data[self.yahoo_symbol]
            
            try:
                market_price = data.get('regularMarketPrice')
                market_change = data.get('regularMarketChange')
                market_change_percent = data.get('regularMarketChangePercent')

                if market_price is None:
                    return

                # 2. 建立一個 "特徵值" (Tuple) 來比較數據是否變動
                # 我們只關心價格和漲跌是否改變，時間戳記 (Timestamp) 不算在內
                current_values = (market_price, market_change)

                # 3. 如果不是強制發送，且數據跟上次一樣，就直接跳過 (不發送)
                if not force_send and self.last_sent_values == current_values:
                    # (可選) Print 出來讓你知道它有在跑，只是被過濾了
                    # print(f"💤 [Yahoo] {self.symbol} 價格未變動，跳過推送")
                    return

                # 更新最後發送的數值
                self.last_sent_values = current_values

                market_change_percent = data.get('regularMarketChangePercent')

                  # 🔥 取得即時市場狀態
                is_open, status_label = get_market_status()

                response = {
                    "symbol": self.symbol,
                    "price": round(float(market_price), 2),
                    "change": round(float(market_change), 2) if market_change else 0,
                    "change_percent": round(float(market_change_percent) * 100, 2) if market_change_percent else 0,
                    "timestamp": time.time(),
                    # 新增狀態欄位
                    "market_status": status_label,  # e.g., "Market Closed"
                    "is_market_open": is_open       # e.g., false
                }

                await self.send(text_data=json.dumps(response))
                
            except Exception as e:
                pass

    async def yahoo_poll(self):
        while self.keep_running:
            await asyncio.sleep(2)
            await self.send_stock_data()