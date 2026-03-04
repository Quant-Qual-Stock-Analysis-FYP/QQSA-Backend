"""日常任务服务：EFS 进化、股票分析等"""

from typing import Dict, List, Any

from markets.models import Stock
from analysis.models import RagDocument, EfsAlphaFactor, EfsAlphaEvaluation, MarketContext
from analysis.config.constants import ETF_SYMBOLS
from analysis.services.analysis import generate_and_store_analysis, build_market_context
from analysis.services.efs_evolution import evolve_factors
from analysis.services.ranking import update_all_rankings


def run_efs_evolution(
    mutate_count: int = 3,
    top_k: int = 1,
    top_m: int = 10,
    min_samples: int = 20,
    window_months: int = 12,
    eval_frequency: str = "biweekly",
    step_days: int = 10,
) -> Dict[str, int]:
    """
    Run EFS factor evolution: re-evaluate all factors, promote top 1, generate 3 new.
    """
    eval_count = EfsAlphaEvaluation.objects.count()
    EfsAlphaEvaluation.objects.all().delete()

    evolution_result = evolve_factors(
        mutate_count=mutate_count,
        top_k=top_k,
        top_m=top_m,
        min_samples=min_samples,
        window_months=window_months,
        eval_frequency=eval_frequency,
        step_days=step_days,
    )
    
    return {
        "cleared_evaluations": eval_count,
        **evolution_result
    }


def run_stock_analysis() -> Dict[str, Any]:
    """
    執行所有股票的分析
    
    Returns:
        dict with analysis results and statistics
    """
    market_context = build_market_context()
    results = []
    errors = []
    
    stocks = list(Stock.objects.all())
    etf_symbols = {s for s in ETF_SYMBOLS}
    etf_stocks = [s for s in stocks if s.symbol in etf_symbols]
    equity_stocks = [s for s in stocks if s.symbol not in etf_symbols]

    # Process ETFs first
    for stock in etf_stocks:
        try:
            RagDocument.objects.filter(stock=stock, is_manual=False).delete()
            results.append(generate_and_store_analysis(stock, market_context=market_context))
        except Exception as exc:
            error_msg = {"symbol": stock.symbol, "error": str(exc)}
            results.append(error_msg)
            errors.append(error_msg)

    # Process equities with market context applied
    for stock in equity_stocks:
        try:
            RagDocument.objects.filter(stock=stock, is_manual=False).delete()
            results.append(generate_and_store_analysis(stock, market_context=market_context))
        except Exception as exc:
            error_msg = {"symbol": stock.symbol, "error": str(exc)}
            results.append(error_msg)
            errors.append(error_msg)
    
    return {
        "total_stocks": len(stocks),
        "etf_count": len(etf_stocks),
        "equity_count": len(equity_stocks),
        "success": len(results) - len(errors),
        "failed": len(errors),
        "results": results,
        "errors": errors
    }


def update_market_context() -> Dict[str, Any]:
    """
    更新市場環境數據
    
    Returns:
        dict with market context data
    """
    market_context = build_market_context()
    
    # 保存到數據庫（只保留最新的一條記錄）
    MarketContext.objects.all().delete()  # 刪除舊記錄
    MarketContext.objects.create(
        market_cycle=market_context.get("market_cycle", "Base"),
        market_score=market_context.get("market_score", 50),
        market_bias=market_context.get("market_bias", 0),
        market_reason=market_context.get("market_reason", []),
    )
    
    return {
        "updated": True,
        "market_context": market_context,
    }


def update_rankings() -> Dict[str, Dict[str, int]]:
    """
    更新所有排名（因子排名和股票排名）
    
    Returns:
        dict with ranking update statistics
    """
    return update_all_rankings()
