# markets/utils.py
from datetime import datetime, time
import pytz

def get_market_status():
    """
    判斷美股現在是否開盤
    回傳: (is_open: bool, label: str)
    """
    nyc_tz = pytz.timezone('America/New_York')
    now_nyc = datetime.now(nyc_tz)
    
    # 1. 判斷是否為週末 (週六=5, 週日=6)
    if now_nyc.weekday() >= 5:
        return False, "Market Closed (Weekend)"

    # 2. 判斷時間是否在 09:30 - 16:00 之間
    current_time = now_nyc.time()
    market_open = time(9, 30)
    market_close = time(16, 0)

    if market_open <= current_time <= market_close:
        return True, "Market Open"
    
    if current_time < market_open:
        return False, "Pre-Market"
    
    return False, "Market Closed"