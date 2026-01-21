from django.urls import path

from .views import (
    AnalysisResultListView,
    AnalysisResultDetailView,
    UploadRagDocumentView,
    UploadRagFundamentalDocumentView,
    PortfolioOptimizationView,
    StockScoresListView,
)

urlpatterns = [
    path("results/", AnalysisResultListView.as_view(), name="analysis-results"),
    path("results/<str:symbol>/", AnalysisResultDetailView.as_view(), name="analysis-result-detail"),
    path("scores/", StockScoresListView.as_view(), name="stock-scores-list"),
    path("documents/", UploadRagDocumentView.as_view(), name="upload-rag-document"),
    path("documents/fundamental/", UploadRagFundamentalDocumentView.as_view(), name="upload-rag-fundamental-document"),
    path("portfolio/", PortfolioOptimizationView.as_view(), name="portfolio-optimization"),
]

