"""
Configuration cho AI Routing Engine
"""
import os
from typing import Optional


class Config:
    """Configuration class"""

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Bandit Parameters
    ALPHA: float = float(os.getenv("ALPHA", "0.1"))
    EPSILON: float = float(os.getenv("EPSILON", "0.1"))

    # Anomaly Detection
    Z_SCORE_THRESHOLD: float = float(os.getenv("Z_SCORE_THRESHOLD", "2.5"))

    # Database (future)
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL", None)


config = Config()

