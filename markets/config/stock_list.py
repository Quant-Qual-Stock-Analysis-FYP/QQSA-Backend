# Stock list for data fetching
STOCKS = [
    # 1. 科技七巨頭 (The Magnificent 7) - 必選，交易量最大
    {"symbol": "AAPL", "name": "Apple Inc."},
    {"symbol": "MSFT", "name": "Microsoft Corp."},
    {"symbol": "GOOGL", "name": "Alphabet Inc."},
    {"symbol": "AMZN", "name": "Amazon.com Inc."},
    {"symbol": "NVDA", "name": "NVIDIA Corp."},
    {"symbol": "META", "name": "Meta Platforms Inc."},
    {"symbol": "TSLA", "name": "Tesla Inc."},
    # 2. 熱門晶片與半導體 (Semiconductors)
    {"symbol": "AMD", "name": "Advanced Micro Devices"},
    {"symbol": "INTC", "name": "Intel Corp."},
    {"symbol": "TSM", "name": "Taiwan Semiconductor"},
    {"symbol": "MU", "name": "Micron Technology"},
    {"symbol": "QCOM", "name": "Qualcomm Inc."},
    # 知名消費品牌 (Consumer Brands) - 大家都認識
    {"symbol": "KO", "name": "Coca-Cola Company"},
    {"symbol": "PEP", "name": "PepsiCo Inc."},
    {"symbol": "MCD", "name": "McDonald's Corp."},
    {"symbol": "SBUX", "name": "Starbucks Corp."},
    {"symbol": "NKE", "name": "NIKE Inc."},
    {"symbol": "DIS", "name": "Walt Disney Company"},
    {"symbol": "NFLX", "name": "Netflix Inc."},
    {"symbol": "COST", "name": "Costco Wholesale"},
    {"symbol": "WMT", "name": "Walmart Inc."},
    # 金融與支付 (Finance)
    {"symbol": "JPM", "name": "JPMorgan Chase & Co."},
    {"symbol": "BAC", "name": "Bank of America Corp."},
    {"symbol": "V", "name": "Visa Inc."},
    {"symbol": "MA", "name": "Mastercard Inc."},
    {"symbol": "PYPL", "name": "PayPal Holdings"},
    {"symbol": "HOOD", "name": "Robinhood Markets"},
    {"symbol": "COIN", "name": "Coinbase Global"},
    # 工業與汽車 (Industrial & Auto)
    {"symbol": "BA", "name": "Boeing Company"},
    {"symbol": "F", "name": "Ford Motor Company"},
    {"symbol": "GM", "name": "General Motors"},
    {"symbol": "GE", "name": "General Electric"},
    {"symbol": "CAT", "name": "Caterpillar Inc."},
    # 軟體與網路服務 (Software & Services)
    {"symbol": "CRM", "name": "Salesforce Inc."},
    {"symbol": "ADBE", "name": "Adobe Inc."},
    {"symbol": "UBER", "name": "Uber Technologies"},
    {"symbol": "ABNB", "name": "Airbnb Inc."},
    {"symbol": "PLTR", "name": "Palantir Technologies"},
    {"symbol": "SPOT", "name": "Spotify Technology"},
    # 醫療與生技 (Healthcare)
    {"symbol": "JNJ", "name": "Johnson & Johnson"},
    {"symbol": "PFE", "name": "Pfizer Inc."},
    {"symbol": "MRNA", "name": "Moderna Inc."},
    # 能源 (Energy)
    {"symbol": "XOM", "name": "Exxon Mobil Corp."},
    {"symbol": "CVX", "name": "Chevron Corp."},
    # 指數 ETF (Market Indices) - 用來顯示大盤走勢
    {"symbol": "SPY", "name": "SPDR S&P 500 ETF"},
    {"symbol": "QQQ", "name": "Invesco QQQ Trust"},
    {"symbol": "DIA", "name": "SPDR Dow Jones ETF"},
    # 債券與防禦 ETF (Bonds/Defensive)
    {"symbol": "TLT", "name": "iShares 20+ Year Treasury Bond ETF"},
    {"symbol": "IEF", "name": "iShares 7-10 Year Treasury Bond ETF"},
    {"symbol": "BIL", "name": "SPDR Bloomberg 1-3 Month T-Bill ETF"},
    # 熱門新題材 (Trending)
    {"symbol": "SMCI", "name": "Super Micro Computer Inc."},
    {"symbol": "ARM", "name": "Arm Holdings"},
    {"symbol": "ASML", "name": "ASML Holding"},
    # 高波動網紅股 (High Volatility/Meme) - Demo 時價格會一直跳
    {"symbol": "GME", "name": "GameStop Corp."},
    {"symbol": "AMC", "name": "AMC Entertainment"},
    {"symbol": "SOFI", "name": "SoFi Technologies"}
]
