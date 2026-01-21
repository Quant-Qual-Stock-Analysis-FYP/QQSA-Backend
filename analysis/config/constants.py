ETF_SYMBOLS = {"SPY", "QQQ", "DIA", "TLT", "IEF", "BIL"}

# ETF 分類用於跨市場分析
EQUITY_ETFS = {"SPY", "QQQ", "DIA"}  # 股票類：進攻/風險資產
BOND_ETFS = {"TLT", "IEF"}  # 債券類：防守/避險資產
CASH_ETFS = {"BIL"}  # 現金類：絕對防守
COMMODITY_ETFS = set()  # 大宗商品：暫時不使用

# 成長型 vs 價值型
GROWTH_ETF = "QQQ"  # 成長與貪婪
VALUE_ETF = "DIA"  # 價值與防禦
BENCHMARK_ETF = "SPY"  # 整體市場基準

# 債券分類
LONG_BOND_ETF = "TLT"  # 20年+美債：對抗衰退
MID_BOND_ETF = "IEF"  # 7-10年美債：基準無風險利率
SHORT_BOND_ETF = "BIL"  # 1-3月短債：現金為王
