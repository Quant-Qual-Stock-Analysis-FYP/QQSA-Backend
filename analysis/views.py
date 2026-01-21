"""API 入口：分析结果、RAG 上传、组合优化等。"""

import logging

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from markets.models import Stock
from .models import AnalysisResult, RagDocument, RagDocumentFundamental
from .services.portfolio import build_sparse_portfolio
from .services.rag import _build_embedding
from .utils.validators import ValidationError, validate_symbol

logger = logging.getLogger(__name__)


class AnalysisResultListView(APIView):
    """返回所有存储的AI分析结果"""

    def get(self, request):
        try:
            # 优化：使用select_related和only减少查询
            results = (
                AnalysisResult.objects.select_related("stock")
                .only("result", "stock_rank", "etf_rank", "stock__symbol")
                .all()
            )
            payload = []
            for res in results:
                try:
                    if not isinstance(res.result, dict):
                        continue
                    result = res.result.copy()
                    result["stock_rank"] = res.stock_rank
                    result["etf_rank"] = res.etf_rank
                    payload.append(result)
                except Exception as e:
                    logger.debug(f"Error processing analysis result {res.id}: {e}")
                    continue
            return Response(payload, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error in AnalysisResultListView: {e}", exc_info=True)
            return Response(
                {"error": "Internal server error"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class AnalysisResultDetailView(APIView):
    """返回特定股票的AI分析结果"""

    def get(self, request, symbol: str):
        # 输入验证
        is_valid, error_msg = validate_symbol(symbol)
        if not is_valid:
            return Response(
                {"error": error_msg or "Invalid symbol"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            stock = Stock.objects.filter(symbol=symbol.upper()).first()
            if not stock:
                return Response(
                    {"error": "stock not found"},
                    status=status.HTTP_404_NOT_FOUND
                )

            res = getattr(stock, "analysis_result", None)
            if res:
                try:
                    if not isinstance(res.result, dict):
                        return Response(
                            {"error": "Invalid analysis data"},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR
                        )
                    result = res.result.copy()
                    result["stock_rank"] = res.stock_rank
                    result["etf_rank"] = res.etf_rank
                    return Response(result, status=status.HTTP_200_OK)
                except Exception as e:
                    logger.error(f"Error processing analysis result: {e}", exc_info=True)
                    return Response(
                        {"error": "Error processing analysis data"},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )

            return Response(
                {"error": "analysis not ready"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(f"Error in AnalysisResultDetailView: {e}", exc_info=True)
            return Response(
                {"error": "Internal server error"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class UploadRagDocumentView(APIView):
    """允许手动上传RAG文档"""

    def post(self, request):
        symbol = request.data.get("symbol")
        content = request.data.get("content")
        source = request.data.get("source", "manual")
        
        # 输入验证
        if not symbol or not content:
            return Response(
                {"error": "symbol and content are required"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        is_valid, error_msg = validate_symbol(symbol)
        if not is_valid:
            return Response(
                {"error": error_msg or "Invalid symbol"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if not isinstance(content, str) or not content.strip():
            return Response(
                {"error": "content must be a non-empty string"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            stock = Stock.objects.filter(symbol=symbol.upper()).first()
            if not stock:
                return Response(
                    {"error": "stock not found"},
                    status=status.HTTP_404_NOT_FOUND
                )

            RagDocument.objects.create(
                stock=stock,
                content=content.strip(),
                source=str(source)[:255] if source else "manual",
                is_manual=True,
                embedding=_build_embedding(content),
            )
            return Response({"status": "ok"}, status=status.HTTP_201_CREATED)
        except Exception as e:
            logger.error(f"Error uploading RAG document: {e}", exc_info=True)
            return Response(
                {"error": "Failed to upload document"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class UploadRagFundamentalDocumentView(APIView):
    """允许手动上传基本面RAG文档"""

    def post(self, request):
        symbol = request.data.get("symbol")
        content = request.data.get("content")
        source = request.data.get("source", "manual")
        
        # 输入验证
        if not symbol or not content:
            return Response(
                {"error": "symbol and content are required"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        is_valid, error_msg = validate_symbol(symbol)
        if not is_valid:
            return Response(
                {"error": error_msg or "Invalid symbol"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if not isinstance(content, str) or not content.strip():
            return Response(
                {"error": "content must be a non-empty string"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            stock = Stock.objects.filter(symbol=symbol.upper()).first()
            if not stock:
                return Response(
                    {"error": "stock not found"},
                    status=status.HTTP_404_NOT_FOUND
                )

            RagDocumentFundamental.objects.create(
                stock=stock,
                content=content.strip(),
                source=str(source)[:255] if source else "manual",
                is_manual=True,
                embedding=_build_embedding(content),
            )
            return Response({"status": "ok"}, status=status.HTTP_201_CREATED)
        except Exception as e:
            logger.error(f"Error uploading fundamental RAG document: {e}", exc_info=True)
            return Response(
                {"error": "Failed to upload document"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class StockScoresListView(APIView):
    """返回所有股票的簡化評分數據"""

    def get(self, request):
        try:
            # 优化：使用select_related和only减少查询
            results = (
                AnalysisResult.objects.select_related("stock")
                .only("result", "stock_rank", "etf_rank", "stock__symbol")
                .all()
            )
            payload = []
            for res in results:
                try:
                    if not isinstance(res.result, dict):
                        continue
                    
                    result = res.result
                    scores = result.get("scores", {})
                    
                    # 提取所需字段
                    payload.append({
                        "stock_symbol": res.stock.symbol,
                        "stock_rank": res.stock_rank,
                        "etf_rank": res.etf_rank,
                        "overall_score": scores.get("overall_score"),
                        "risk_score": scores.get("stability"),  
                        "sentiment_score": scores.get("sentiment"),
                        "technical_score": scores.get("technical"),
                        "fundamental_score": scores.get("fundamental"), 
                    })
                except Exception as e:
                    logger.debug(f"Error processing analysis result {res.id}: {e}")
                    continue
            
            return Response(payload, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error in StockScoresListView: {e}", exc_info=True)
            return Response(
                {"error": "Internal server error"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class PortfolioOptimizationView(APIView):
    """稀疏组合优化：基于overall_score和风险惩罚"""

    def post(self, request):
        # 请求参数：top_n（必填），horizon/use_hedge（可选）
        top_n = request.data.get("top_n")
        horizon = request.data.get("horizon", "mid")
        use_hedge = request.data.get("use_hedge", True)
        risk_level = request.data.get("risk_level")

        # 输入验证
        if top_n is None:
            return Response(
                {"error": "top_n is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            top_n_val = int(top_n)
        except (TypeError, ValueError):
            return Response(
                {"error": "top_n must be integer"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            payload = build_sparse_portfolio(
                top_n=top_n_val,
                horizon=horizon,
                use_hedge=bool(use_hedge),
                risk_level=risk_level,
            )
            return Response(payload, status=status.HTTP_200_OK)
        except ValidationError as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            logger.error(f"Error in PortfolioOptimizationView: {e}", exc_info=True)
            return Response(
                {"error": "Internal server error"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

