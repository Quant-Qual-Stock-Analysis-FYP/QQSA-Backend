"""分析模块配置管理：统一管理所有配置参数。"""

from typing import Dict, Any


class PortfolioConfig:
    """组合优化配置"""
    
    # 时间窗口配置
    HORIZON_LOOKBACK: Dict[str, int] = {
        "short": 30,
        "mid": 60,
        "long": 120,
    }
    
    # 风险惩罚参数
    RISK_LAMBDA: Dict[str, int] = {
        "short": 220,
        "mid": 300,
        "long": 380,
    }
    
    CORR_LAMBDA: Dict[str, int] = {
        "short": 18,
        "mid": 15,
        "long": 12,
    }
    
    # 风险偏好调整系数
    RISK_PROFILE_MULTIPLIERS: Dict[str, Dict[str, float]] = {
        "high": {"risk": 0.7, "corr": 0.8},
        "mid": {"risk": 1.0, "corr": 1.0},
        "low": {"risk": 1.3, "corr": 1.2},
    }
    
    # 权重限制
    MAX_WEIGHT: Dict[str, float] = {
        "high": 0.25,
        "mid": 0.2,
        "low": 0.15,
    }
    
    # 对冲配置
    HEDGE_RATIO_BASE = 0.1
    HEDGE_RATIO_HIGH_VOL = 0.3
    HEDGE_RATIO_CRISIS = 0.4
    MARKET_VOL_THRESHOLD_HIGH = 0.2
    MARKET_VOL_THRESHOLD_CRISIS = 0.3
    
    # 行业集中度限制
    SECTOR_CAP_RATIO = 0.4
    
    # 优化参数
    OPTIMIZATION_ITERATIONS = 200
    OPTIMIZATION_STEP_SIZE = 0.01
    RISK_AVERSION: Dict[str, float] = {
        "high": 0.7,
        "mid": 1.0,
        "low": 1.3,
    }
    
    # 数据验证
    MIN_RETURNS_LENGTH = 5
    MIN_RETURNS_RATIO = 0.5  # lookback // 2
    MIN_ABSOLUTE_RETURNS = 10


class EfsConfig:
    """EFS factor evolution config."""

    # Per-run: 3 new factors via LLM mutation from Alpha158 base
    DEFAULT_MUTATE_COUNT = 3
    DEFAULT_TOP_K = 1  # Single active factor for technique score / LLM analysis
    LLM_TEMPERATURE = 0.4
    
    # 因子评估
    DEFAULT_TOP_M = 10
    DEFAULT_FUTURE_DAYS = 20
    DEFAULT_MIN_SAMPLES = 20
    DEFAULT_WINDOW_MONTHS = 12
    DEFAULT_EVAL_FREQUENCY = "biweekly"
    DEFAULT_STEP_DAYS = 10
    
    # 评分权重
    IC_WEIGHT = 0.6
    TOP_RETURN_WEIGHT = 0.4
    
    # 数据验证
    MIN_EVAL_DATES = 1
    MIN_UNIVERSE_SIZE = 20


class CacheConfig:
    """缓存配置"""
    
    # 缓存超时时间（秒）
    FEATURE_SNAPSHOT_TIMEOUT = 3600  # 1小时
    RETURNS_CACHE_TIMEOUT = 1800  # 30分钟
    ANALYSIS_RESULT_TIMEOUT = 3600  # 1小时
    
    # 缓存键前缀
    KEY_PREFIX_FEATURE = "efs:feature:"
    KEY_PREFIX_RETURNS = "portfolio:returns:"
    KEY_PREFIX_ANALYSIS = "analysis:result:"


class ValidationConfig:
    """数据验证配置"""
    
    # 组合优化验证
    MIN_TOP_N = 1
    MAX_TOP_N = 50
    VALID_HORIZONS = {"short", "mid", "long"}
    VALID_RISK_LEVELS = {"high", "mid", "low"}
    
    # 数值验证
    MIN_WEIGHT = 0.0
    MAX_WEIGHT = 1.0
    MIN_SCORE = 0
    MAX_SCORE = 100
    
    # 字符串长度限制
    MAX_SYMBOL_LENGTH = 10
    MAX_SECTOR_LENGTH = 100


def get_portfolio_config() -> Dict[str, Any]:
    """获取组合优化配置"""
    return {
        "horizon_lookback": PortfolioConfig.HORIZON_LOOKBACK,
        "risk_lambda": PortfolioConfig.RISK_LAMBDA,
        "corr_lambda": PortfolioConfig.CORR_LAMBDA,
        "risk_profile_multipliers": PortfolioConfig.RISK_PROFILE_MULTIPLIERS,
        "max_weight": PortfolioConfig.MAX_WEIGHT,
        "hedge_ratio_base": PortfolioConfig.HEDGE_RATIO_BASE,
        "sector_cap_ratio": PortfolioConfig.SECTOR_CAP_RATIO,
    }


def get_efs_config() -> Dict[str, Any]:
    """EFS config for API / admin."""
    return {
        "mutate_count": EfsConfig.DEFAULT_MUTATE_COUNT,
        "top_k": EfsConfig.DEFAULT_TOP_K,
        "top_m": EfsConfig.DEFAULT_TOP_M,
        "future_days": EfsConfig.DEFAULT_FUTURE_DAYS,
        "min_samples": EfsConfig.DEFAULT_MIN_SAMPLES,
        "window_months": EfsConfig.DEFAULT_WINDOW_MONTHS,
        "ic_weight": EfsConfig.IC_WEIGHT,
        "top_return_weight": EfsConfig.TOP_RETURN_WEIGHT,
    }
