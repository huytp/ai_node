"""
Data models cho AI Routing Engine
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class NodeMetrics(BaseModel):
    """Metrics từ node heartbeat"""
    node: str = Field(..., description="Địa chỉ node (0x...)")
    latency: float = Field(..., ge=0, description="Latency (ms)")
    loss: float = Field(..., ge=0, le=1, description="Packet loss rate [0-1]")
    jitter: float = Field(..., ge=0, description="Jitter (ms)")
    uptime: int = Field(..., ge=0, description="Uptime (seconds)")
    bandwidth: float = Field(..., ge=0, description="Bandwidth (Mbps)")
    geo_distance: Optional[float] = Field(None, ge=0, description="Geo distance (km)")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)


class RouteRequest(BaseModel):
    """Request để chọn route"""
    user_location: Optional[str] = Field(None, description="User location")
    time_bucket: Optional[int] = Field(None, description="Time bucket (0-23 for hour)")
    available_nodes: Optional[List[str]] = Field(None, description="List available node addresses")


class RouteResponse(BaseModel):
    """Response với route được chọn"""
    entry: str = Field(..., description="Entry node address")
    exit: str = Field(..., description="Exit node address")
    score: float = Field(..., ge=0, le=1, description="Route quality score [0-1]")


class ReputationUpdate(BaseModel):
    """Reputation update từ anomaly detection"""
    node: str = Field(..., description="Node address")
    score: int = Field(..., ge=0, le=100, description="Reputation score [0-100]")
    z_score: float = Field(..., description="Z-score từ anomaly detection")
    is_anomaly: bool = Field(..., description="Có phải anomaly không")


class URLCheckRequest(BaseModel):
    """Request để kiểm tra URL độc hại"""
    url: str = Field(..., description="URL cần kiểm tra")


class URLCheckResponse(BaseModel):
    """Response từ URL check"""
    url: str = Field(..., description="URL đã kiểm tra")
    is_malicious: bool = Field(..., description="URL có độc hại không")
    probability: float = Field(..., ge=0, le=1, description="Xác suất URL độc hại [0-1]")
    confidence: str = Field(..., description="Độ tin cậy: low, medium, high")

