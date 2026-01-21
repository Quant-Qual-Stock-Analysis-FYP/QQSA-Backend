"""分析模块模型定义（RAG、EFS、AI 结果缓存等）。"""

from django.db import models
from django.contrib.postgres.fields import ArrayField

from markets.models import Stock


class RagDocument(models.Model):
    """Vectorized text snippets used for RAG sentiment."""

    stock = models.ForeignKey(
        Stock,
        on_delete=models.CASCADE,
        related_name="rag_documents",
    )
    content = models.TextField()
    source = models.CharField(max_length=255, blank=True)
    is_manual = models.BooleanField(default=False)
    embedding = ArrayField(models.FloatField(), default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["stock", "created_at"]),
        ]

    def __str__(self) -> str:
        return f"{self.stock.symbol} RAG #{self.id}"

class RagDocumentFundamental(models.Model):
    """Vectorized text snippets used for RAG fundamentals."""

    stock = models.ForeignKey(
        Stock,
        on_delete=models.CASCADE,
        related_name="rag_fundamentals",
    )
    content = models.TextField()
    source = models.CharField(max_length=255, blank=True)
    is_manual = models.BooleanField(default=False)
    embedding = ArrayField(models.FloatField(), default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["stock", "created_at"]),
        ]

    def __str__(self) -> str:
        return f"{self.stock.symbol} RAG Fundamental #{self.id}"

class EfsDataPoint(models.Model):
    """Historical OHLCV required by the EFS (time-series) engine."""

    stock = models.ForeignKey(
        Stock,
        on_delete=models.CASCADE,
        related_name="efs_data",
    )
    date = models.DateField()
    open = models.DecimalField(max_digits=14, decimal_places=4)
    high = models.DecimalField(max_digits=14, decimal_places=4)
    low = models.DecimalField(max_digits=14, decimal_places=4)
    close = models.DecimalField(max_digits=14, decimal_places=4)
    volume = models.BigIntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("stock", "date")
        indexes = [
            models.Index(fields=["stock", "date"]),
        ]

    def __str__(self) -> str:
        return f"{self.stock.symbol} {self.date}"

class EfsAlphaFactor(models.Model):
    """LLM-generated alpha factor used in EFS evolution."""

    name = models.CharField(max_length=128)
    description = models.TextField(blank=True)
    expression = models.TextField()
    status = models.CharField(max_length=32, default="candidate")
    last_score = models.FloatField(null=True, blank=True)
    # 排名字段（基于 last_score）
    factor_rank = models.PositiveIntegerField(null=True, blank=True, help_text="因子排名（1-based，基于 last_score）")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-updated_at", "-created_at")
        indexes = [
            models.Index(fields=["status", "updated_at"]),
            models.Index(fields=["factor_rank"]),
        ]

    def __str__(self) -> str:
        return f"EFS alpha {self.name}"

class EfsAlphaEvaluation(models.Model):
    """Per-run evaluation snapshot for an alpha factor."""

    factor = models.ForeignKey(
        EfsAlphaFactor,
        on_delete=models.CASCADE,
        related_name="evaluations",
    )
    as_of = models.DateField()
    ic = models.FloatField(null=True, blank=True)
    top_return = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-as_of", "-created_at")
        indexes = [
            models.Index(fields=["factor", "as_of"]),
        ]

class MarketContext(models.Model):
    """Market context data (cycle, bias, etc.)"""

    market_cycle = models.CharField(max_length=32, default="Base")  # Bull, Bear, Base
    market_score = models.IntegerField(default=50)  # 0-100
    market_bias = models.IntegerField(default=0)  # -8, 0, 8
    market_reason = ArrayField(models.TextField(), default=list, blank=True)  # List of reason strings
    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-updated_at",)
        indexes = [
            models.Index(fields=["-updated_at"]),
        ]

    def __str__(self) -> str:
        return f"Market Context ({self.market_cycle}) - {self.updated_at.strftime('%Y-%m-%d %H:%M')}"

    def to_dict(self) -> dict:
        """Convert to dictionary format matching API response"""
        return {
            "market_cycle": self.market_cycle,
            "market_score": self.market_score,
            "market_bias": self.market_bias,
            "market_reason": self.market_reason,
        }

class AnalysisResult(models.Model):
    """Persisted AI output per stock for quick retrieval."""

    stock = models.OneToOneField(
        Stock,
        on_delete=models.CASCADE,
        related_name="analysis_result",
    )
    result = models.JSONField(default=dict, blank=True)
    # 股票排名字段（仅用于股票）
    stock_rank = models.PositiveIntegerField(null=True, blank=True, help_text="股票排名（1-based，基于 overall_score，仅用于股票）")
    # ETF 排名字段（仅用于 ETF）
    etf_rank = models.PositiveIntegerField(null=True, blank=True, help_text="ETF排名（1-based，基于 overall_score，仅用于ETF）")
    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["stock_rank"]),
            models.Index(fields=["etf_rank"]),
        ]

    def __str__(self) -> str:
        return f"{self.stock.symbol} analysis"

