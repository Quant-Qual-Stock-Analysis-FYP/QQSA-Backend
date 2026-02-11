"""API 入口：分析结果、RAG 上传、组合优化等。"""

import logging

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from markets.models import Stock
from .models import AnalysisResult, RagDocument, RagDocumentFundamental, MarketContext
from .services.portfolio import build_sparse_portfolio
from .services.rag import _build_embedding
from .services.analysis import build_market_context
from .utils.validators import ValidationError, validate_symbol
from .utils.document_parser import extract_text_from_file

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
    """统一的上传RAG文档视图（支持文本内容或文件上传：docx, pdf）
    
    支持两种请求方式：
    1. multipart/form-data: 上传文件时使用，字段：symbol, file
    2. application/json: 发送文本内容时使用，字段：symbol, content
    
    通过 doc_type 参数区分保存到不同表：
    - "sentiment" 或默认：保存到 RagDocument
    - "fundamental"：保存到 RagDocumentFundamental
    """

    def post(self, request, **kwargs):
        file = request.FILES.get("file")
        
        # 从 URL kwargs 获取 doc_type，默认为 "sentiment"
        doc_type = self.kwargs.get("doc_type", "sentiment")
        doc_type = doc_type.lower() if doc_type else "sentiment"
        if doc_type not in ["sentiment", "fundamental"]:
            return Response(
                {"error": "doc_type must be 'sentiment' or 'fundamental'"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # 根据是否有文件上传，决定处理方式
        if file:
            # 文件上传模式：使用 multipart/form-data
            symbol = request.data.get("symbol")
            
            if not symbol:
                return Response(
                    {"error": "symbol is required"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            is_valid, error_msg = validate_symbol(symbol)
            if not is_valid:
                return Response(
                    {"error": error_msg or "Invalid symbol"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # 验证文件类型
            filename = file.name.lower()
            if not (filename.endswith('.docx') or filename.endswith('.pdf')):
                return Response(
                    {"error": "file must be .docx or .pdf"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # 提取文本内容
            try:
                extracted_text = extract_text_from_file(file, file.name)
                if not extracted_text or not extracted_text.strip():
                    return Response(
                        {"error": "failed to extract text from file or file is empty"},
                        status=status.HTTP_400_BAD_REQUEST
                    )
                content = extracted_text.strip()
            except Exception as e:
                logger.error(f"Error extracting text from file: {e}", exc_info=True)
                return Response(
                    {"error": "Failed to extract text from file"},
                    status=status.HTTP_400_BAD_REQUEST
                )
        else:
            # 文本内容模式：使用 application/json
            symbol = request.data.get("symbol")
            content = request.data.get("content")
            
            if not symbol:
                return Response(
                    {"error": "symbol is required"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            if not content:
                return Response(
                    {"error": "content is required when no file is uploaded"},
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
            
            content = content.strip()
        
        # 保存到数据库
        try:
            stock = Stock.objects.filter(symbol=symbol.upper()).first()
            if not stock:
                return Response(
                    {"error": "stock not found"},
                    status=status.HTTP_404_NOT_FOUND
                )

            # 根据 doc_type 选择对应的模型
            # 如果有文件上传，保存文件名；否则filename为None
            filename = file.name if file else None
            
            if doc_type == "fundamental":
                RagDocumentFundamental.objects.create(
                    stock=stock,
                    content=content,
                    source="manual",
                    is_manual=True,
                    filename=filename,
                    embedding=_build_embedding(content),
                )
            else:  # sentiment (default)
                RagDocument.objects.create(
                    stock=stock,
                    content=content,
                    source="manual",
                    is_manual=True,
                    filename=filename,
                    embedding=_build_embedding(content),
                )
            
            return Response({"status": "ok"}, status=status.HTTP_201_CREATED)
        except Exception as e:
            logger.error(f"Error uploading RAG document: {e}", exc_info=True)
            return Response(
                {"error": "Failed to upload document"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ListManualRagDocumentsView(APIView):
    """列出手动上传的RAG文档（仅返回有文件名的文档）
    
    通过 doc_type 参数区分：
    - "sentiment"：返回 RagDocument 中的手动上传文件
    - "fundamental"：返回 RagDocumentFundamental 中的手动上传文件
    """

    def get(self, request, **kwargs):
        # 从 URL kwargs 获取 doc_type
        doc_type = self.kwargs.get("doc_type", "sentiment")
        doc_type = doc_type.lower() if doc_type else "sentiment"
        
        if doc_type not in ["sentiment", "fundamental"]:
            return Response(
                {"error": "doc_type must be 'sentiment' or 'fundamental'"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # 根据 doc_type 选择对应的模型
            if doc_type == "fundamental":
                documents = RagDocumentFundamental.objects.filter(
                    is_manual=True,
                    filename__isnull=False
                ).select_related("stock").order_by("-created_at")
            else:  # sentiment
                documents = RagDocument.objects.filter(
                    is_manual=True,
                    filename__isnull=False
                ).select_related("stock").order_by("-created_at")
            
            payload = []
            for doc in documents:
                payload.append({
                    "id": doc.id,
                    "symbol": doc.stock.symbol,
                    "filename": doc.filename,
                    "source": doc.source,
                    "created_at": doc.created_at.isoformat(),
                })
            
            return Response(payload, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error listing manual RAG documents: {e}", exc_info=True)
            return Response(
                {"error": "Failed to list documents"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class DeleteManualRagDocumentView(APIView):
    """删除手动上传的RAG文档
    
    通过 doc_type 参数区分：
    - "sentiment"：删除 RagDocument 中的文档
    - "fundamental"：删除 RagDocumentFundamental 中的文档
    """

    def delete(self, request, doc_id, **kwargs):
        # 从 URL kwargs 获取 doc_type
        doc_type = self.kwargs.get("doc_type", "sentiment")
        doc_type = doc_type.lower() if doc_type else "sentiment"
        
        if doc_type not in ["sentiment", "fundamental"]:
            return Response(
                {"error": "doc_type must be 'sentiment' or 'fundamental'"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # 根据 doc_type 选择对应的模型
            if doc_type == "fundamental":
                document = RagDocumentFundamental.objects.filter(
                    id=doc_id,
                    is_manual=True
                ).first()
            else:  # sentiment
                document = RagDocument.objects.filter(
                    id=doc_id,
                    is_manual=True
                ).first()
            
            if not document:
                return Response(
                    {"error": "Document not found or not a manual upload"},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            document.delete()
            return Response({"status": "ok"}, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error deleting manual RAG document: {e}", exc_info=True)
            return Response(
                {"error": "Failed to delete document"},
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


class MarketContextView(APIView):
    """返回市場環境數據"""

    def get(self, request):
        try:
            # 嘗試從數據庫獲取最新的 market_context
            market_context_obj = MarketContext.objects.first()
            
            if market_context_obj:
                # 從數據庫返回
                payload = market_context_obj.to_dict()
            else:
                # 如果數據庫中沒有，則構建新的（作為後備）
                market_context = build_market_context()
                payload = market_context
            
            return Response(payload, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error in MarketContextView: {e}", exc_info=True)
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

