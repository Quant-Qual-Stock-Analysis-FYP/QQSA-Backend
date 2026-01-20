from django.db import models

class Stock(models.Model):
    symbol = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=100)
    sector = models.CharField(max_length=50, blank=True)
    
    # ✅ 優化重點：使用 JSONField 儲存所有基本面數據
    # 這樣無論 AAPL 和 NVDA 的欄位有什麼不同，通通都能存進去
    details = models.JSONField(default=dict, blank=True) 
    
    # 記錄最後一次更新基本面數據的時間
    last_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.symbol

class StockNews(models.Model):
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name='news')
    title = models.CharField(max_length=500)
    summary = models.TextField(blank=True, null=True)  # ✅ 新增：摘要
    link = models.URLField(max_length=1000)
    publisher = models.CharField(max_length=200, blank=True, null=True)
    icon_url = models.URLField(max_length=1000, blank=True, null=True)
    publish_time = models.DateTimeField(null=True, blank=True) # ✅ 新增：原始發布時間
    created_at = models.DateTimeField(auto_now_add=True) # 我們抓取的時間

    def __str__(self):
        return f"{self.stock.symbol}: {self.title[:30]}..."