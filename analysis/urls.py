from django.urls import path

from .views import (
    AnalysisResultListView,
    AnalysisResultDetailView,
    UploadRagDocumentView,
    ListManualRagDocumentsView,
    DeleteManualRagDocumentView,
    PortfolioOptimizationView,
    StockScoresListView,
    MarketContextView,
)

urlpatterns = [
    path("results/", AnalysisResultListView.as_view(), name="analysis-results"),
    path("results/<str:symbol>/", AnalysisResultDetailView.as_view(), name="analysis-result-detail"),
    path("scores/", StockScoresListView.as_view(), name="stock-scores-list"),
    path("market-context/", MarketContextView.as_view(), name="market-context"),
    path("documents/sentiment/", UploadRagDocumentView.as_view(), {"doc_type": "sentiment"}, name="upload-rag-sentiment-document"),
    path("documents/fundamental/", UploadRagDocumentView.as_view(), {"doc_type": "fundamental"}, name="upload-rag-fundamental-document"),
    path("documents/sentiment/list/", ListManualRagDocumentsView.as_view(), {"doc_type": "sentiment"}, name="list-rag-sentiment-documents"),
    path("documents/fundamental/list/", ListManualRagDocumentsView.as_view(), {"doc_type": "fundamental"}, name="list-rag-fundamental-documents"),
    path("documents/sentiment/<int:doc_id>/", DeleteManualRagDocumentView.as_view(), {"doc_type": "sentiment"}, name="delete-rag-sentiment-document"),
    path("documents/fundamental/<int:doc_id>/", DeleteManualRagDocumentView.as_view(), {"doc_type": "fundamental"}, name="delete-rag-fundamental-document"),
    path("portfolio/", PortfolioOptimizationView.as_view(), name="portfolio-optimization"),
]

