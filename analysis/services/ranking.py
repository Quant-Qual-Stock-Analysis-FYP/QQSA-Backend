"""排名更新服务：更新因子排名和股票排名"""

import logging
from typing import Dict

from django.db import transaction

from analysis.config.constants import ETF_SYMBOLS
from analysis.models import AnalysisResult, EfsAlphaFactor

logger = logging.getLogger(__name__)


def update_factor_rankings() -> Dict[str, int]:
    """更新所有因子的排名（基于 last_score）"""
    try:
        # 优化：只查询需要的字段
        factors = list(
            EfsAlphaFactor.objects.exclude(last_score__isnull=True)
            .order_by("-last_score")
            .only("id", "factor_rank", "last_score")
        )
        
        if not factors:
            return {"updated": 0}
        
        updated_count = 0
        
        try:
            with transaction.atomic():
                for rank, factor in enumerate(factors, start=1):
                    if factor.factor_rank != rank:
                        factor.factor_rank = rank
                        factor.save(update_fields=["factor_rank"])
                        updated_count += 1
        except Exception as e:
            logger.error(f"Error updating factor rankings: {e}", exc_info=True)
            return {"updated": 0, "error": str(e)}
        
        return {"updated": updated_count}
    except Exception as e:
        logger.error(f"Error in update_factor_rankings: {e}", exc_info=True)
        return {"updated": 0, "error": str(e)}


def update_stock_rankings() -> Dict[str, int]:
    """更新所有股票的排名（基于 overall_score），分别计算 stock_rank 和 etf_rank"""
    try:
        # 优化：使用select_related和only减少查询
        results = list(
            AnalysisResult.objects.all()
            .select_related("stock")
            .only("id", "result", "stock_rank", "etf_rank", "stock__symbol")
        )
        
        if not results:
            return {"updated": 0}
        
        # 分离股票和 ETF
        stock_results = []
        etf_results = []
        
        for res in results:
            try:
                if not isinstance(res.result, dict):
                    continue
                
                # 支持新格式 (scores.overall_score) 和旧格式 (overall_score)
                scores = res.result.get("scores", {})
                overall_score = scores.get("overall_score") or res.result.get("overall_score")
                
                if overall_score is None:
                    continue
                
                if not res.stock or not res.stock.symbol:
                    continue
                
                symbol = res.stock.symbol.upper()
                score_int = int(overall_score)
                
                if symbol in ETF_SYMBOLS:
                    etf_results.append((res, score_int))
                else:
                    stock_results.append((res, score_int))
            except Exception as e:
                logger.debug(f"Error processing result {res.id}: {e}")
                continue
        
        # 分别排序
        stock_results.sort(key=lambda x: x[1], reverse=True)
        etf_results.sort(key=lambda x: x[1], reverse=True)
        
        updated_count = 0
        
        try:
            with transaction.atomic():
                # 更新股票排名
                for rank, (result, _) in enumerate(stock_results, start=1):
                    if result.stock_rank != rank:
                        result.stock_rank = rank
                        result.etf_rank = None  # 股票没有 ETF 排名
                        result.save(update_fields=["stock_rank", "etf_rank"])
                        updated_count += 1
                
                # 更新 ETF 排名
                for rank, (result, _) in enumerate(etf_results, start=1):
                    if result.etf_rank != rank:
                        result.etf_rank = rank
                        result.stock_rank = None  # ETF 没有股票排名
                        result.save(update_fields=["etf_rank", "stock_rank"])
                        updated_count += 1
        except Exception as e:
            logger.error(f"Error updating stock rankings: {e}", exc_info=True)
            return {"updated": 0, "error": str(e)}
        
        return {"updated": updated_count}
    except Exception as e:
        logger.error(f"Error in update_stock_rankings: {e}", exc_info=True)
        return {"updated": 0, "error": str(e)}


def update_all_rankings() -> Dict[str, Dict[str, int]]:
    """
    更新所有排名（因子排名和股票排名）
    
    Returns:
        dict with both ranking update statistics
    """
    factor_result = update_factor_rankings()
    stock_result = update_stock_rankings()
    
    return {
        "factors": factor_result,
        "stocks": stock_result
    }
