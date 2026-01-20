"""股票数据更新服务：从外部 API 获取股票详情和新闻"""

import finnhub
import datetime
import re
from typing import Dict, List
from django.conf import settings
from django.utils.timezone import make_aware
from yahooquery import Ticker

from markets.models import Stock, StockNews
from markets.config.stock_list import STOCKS


def update_stock_details() -> Dict[str, any]:
    """
    使用 yahooquery 獲取真實股票基本面數據
    
    Returns:
        dict with statistics: {"success": int, "failed": int, "total": int, "errors": List[str]}
    """
    symbols = [stock['symbol'] for stock in STOCKS]
    success_count = 0
    failed_count = 0
    errors = []

    try:
        # 初始化 Ticker
        tickers = Ticker(symbols, asynchronous=True)
        
        # 獲取資料
        data = tickers.get_modules('summaryProfile summaryDetail price defaultKeyStatistics')

        # 處理回傳的資料
        for symbol in symbols:
            if symbol not in data or isinstance(data[symbol], str):
                failed_count += 1
                errors.append(f"{symbol}: 無法獲取數據")
                continue

            try:
                stock_data = data[symbol]
                
                profile = stock_data.get('summaryProfile', {})
                detail = stock_data.get('summaryDetail', {})
                price_info = stock_data.get('price', {})
                stats = stock_data.get('defaultKeyStatistics', {})

                # 存入資料庫
                stock, created = Stock.objects.get_or_create(symbol=symbol)
                
                stock.name = price_info.get('shortName') or price_info.get('longName') or symbol
                stock.sector = profile.get('sector', 'Unknown')
                
                full_details = {
                    **profile,
                    **detail,
                    **price_info,
                    **stats
                }
                
                stock.details = full_details
                stock.save()
                success_count += 1
                
            except Exception as e:
                failed_count += 1
                errors.append(f"{symbol}: {str(e)}")

    except Exception as e:
        errors.append(f"整體錯誤: {str(e)}")
        failed_count = len(symbols)

    return {
        "success": success_count,
        "failed": failed_count,
        "total": len(symbols),
        "errors": errors[:10]  # 只保留前10個錯誤
    }


def _generate_keywords(symbol: str, full_name: str) -> set:
    """產生過濾用的關鍵字"""
    keywords = set()
    keywords.add(symbol.lower())
    
    clean_name = full_name.lower()
    suffixes = [
        r",? inc\.?", r",? corp\.?", r",? corporation", r",? ltd\.?", 
        r",? plc", r",? s\.a\.", r",? holdings?", r",? group", r"\.com"
    ]
    for suffix in suffixes:
        clean_name = re.sub(suffix, "", clean_name)
    
    clean_name = clean_name.strip()
    if len(clean_name) > 1:
        keywords.add(clean_name)
    
    return keywords


def update_stock_news(days_back: int = 3) -> Dict[str, any]:
    """
    抓取股票新聞並存入資料庫
    
    Args:
        days_back: 抓取過去幾天的新聞（預設3天）
    
    Returns:
        dict with statistics: {"total_stocks": int, "success": int, "failed": int, "total_news": int, "errors": List[str]}
    """
    finnhub_client = finnhub.Client(api_key=settings.FINNHUB_API_KEY)
    stocks = Stock.objects.all()
    
    if not stocks:
        return {
            "total_stocks": 0,
            "success": 0,
            "failed": 0,
            "total_news": 0,
            "errors": ["資料庫為空，請先執行 update_details"]
        }

    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=days_back)
    _from = start_date.strftime("%Y-%m-%d")
    _to = today.strftime("%Y-%m-%d")

    success_count = 0
    failed_count = 0
    total_news_count = 0
    errors = []

    for stock in stocks:
        keywords = _generate_keywords(stock.symbol, stock.name)

        try:
            # 呼叫 Finnhub API
            news_list = finnhub_client.company_news(stock.symbol, _from=_from, to=_to)
            
            if not news_list:
                continue

            # 清除舊新聞
            StockNews.objects.filter(stock=stock).delete()

            count = 0
            saved_headlines = set()

            for item in news_list:
                headline = item.get('headline', '').strip()
                summary = item.get('summary', '').strip()
                url = item.get('url', '')
                publisher = item.get('source', 'Finnhub')
                image = item.get('image', '')
                timestamp = item.get('datetime', 0)

                if not headline or not url:
                    continue

                # 嚴格過濾邏輯
                full_text = (headline + " " + summary).lower()
                is_relevant = False
                for kw in keywords:
                    if kw in full_text:
                        is_relevant = True
                        break
                
                if not is_relevant:
                    continue
                
                if headline in saved_headlines:
                    continue

                # 處理發布時間
                publish_dt = None
                if timestamp:
                    try:
                        dt = datetime.datetime.fromtimestamp(timestamp)
                        publish_dt = make_aware(dt)
                    except Exception:
                        publish_dt = None

                # 特殊處理: 如果發布商是 Yahoo，不存圖片
                if publisher and 'yahoo' in publisher.lower():
                    final_icon_url = None
                else:
                    final_icon_url = image if image else None

                # 存入資料庫
                StockNews.objects.create(
                    stock=stock,
                    title=headline,
                    summary=summary,
                    link=url,
                    publisher=publisher,
                    icon_url=final_icon_url,
                    publish_time=publish_dt
                )
                saved_headlines.add(headline)
                count += 1
                
                if count >= 10:
                    break

            total_news_count += count
            success_count += 1

        except Exception as e:
            failed_count += 1
            errors.append(f"{stock.symbol}: {str(e)}")

    return {
        "total_stocks": stocks.count(),
        "success": success_count,
        "failed": failed_count,
        "total_news": total_news_count,
        "errors": errors[:10]
    }
