"""数据验证工具：提供统一的验证函数。"""

from typing import Optional, Tuple

from ..config.settings import ValidationConfig


class ValidationError(Exception):
    """验证错误异常"""
    pass


def validate_top_n(top_n: int) -> Tuple[bool, Optional[str]]:
    """验证top_n参数"""
    if not isinstance(top_n, int):
        return False, "top_n must be an integer"
    if top_n < ValidationConfig.MIN_TOP_N:
        return False, f"top_n must be at least {ValidationConfig.MIN_TOP_N}"
    if top_n > ValidationConfig.MAX_TOP_N:
        return False, f"top_n must be at most {ValidationConfig.MAX_TOP_N}"
    return True, None


def validate_horizon(horizon: str) -> Tuple[bool, Optional[str]]:
    """验证horizon参数"""
    if not isinstance(horizon, str):
        return False, "horizon must be a string"
    horizon_lower = horizon.lower().strip()
    if horizon_lower not in ValidationConfig.VALID_HORIZONS:
        return False, f"horizon must be one of {ValidationConfig.VALID_HORIZONS}"
    return True, None


def validate_risk_level(risk_level: Optional[str]) -> Tuple[bool, Optional[str]]:
    """验证risk_level参数"""
    if risk_level is None:
        return True, None
    if not isinstance(risk_level, str):
        return False, "risk_level must be a string"
    risk_lower = risk_level.lower().strip()
    if risk_lower not in ValidationConfig.VALID_RISK_LEVELS:
        return False, f"risk_level must be one of {ValidationConfig.VALID_RISK_LEVELS}"
    return True, None


def validate_weight(weight: float) -> Tuple[bool, Optional[str]]:
    """验证权重值"""
    if not isinstance(weight, (int, float)):
        return False, "weight must be a number"
    if weight < ValidationConfig.MIN_WEIGHT:
        return False, f"weight must be at least {ValidationConfig.MIN_WEIGHT}"
    if weight > ValidationConfig.MAX_WEIGHT:
        return False, f"weight must be at most {ValidationConfig.MAX_WEIGHT}"
    return True, None


def validate_score(score: int) -> Tuple[bool, Optional[str]]:
    """验证分数值"""
    if not isinstance(score, int):
        return False, "score must be an integer"
    if score < ValidationConfig.MIN_SCORE:
        return False, f"score must be at least {ValidationConfig.MIN_SCORE}"
    if score > ValidationConfig.MAX_SCORE:
        return False, f"score must be at most {ValidationConfig.MAX_SCORE}"
    return True, None


def validate_symbol(symbol: str) -> Tuple[bool, Optional[str]]:
    """验证股票代码"""
    if not isinstance(symbol, str):
        return False, "symbol must be a string"
    if not symbol.strip():
        return False, "symbol cannot be empty"
    if len(symbol) > ValidationConfig.MAX_SYMBOL_LENGTH:
        return False, f"symbol length must be at most {ValidationConfig.MAX_SYMBOL_LENGTH}"
    return True, None
