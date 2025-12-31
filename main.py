"""
AI Routing Engine - FastAPI Application
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional
import uvicorn

from models import RouteRequest, RouteResponse, NodeMetrics, ReputationUpdate, URLCheckRequest, URLCheckResponse
from bandit import ContextualBandit
from anomaly_detection import AnomalyDetector
from url_detector import url_detector

app = FastAPI(
    title="AI Routing Engine",
    description="Contextual Multi-Armed Bandit cho route selection",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
bandit = ContextualBandit(alpha=0.1, epsilon=0.1)
anomaly_detector = AnomalyDetector(threshold=2.5)

# In-memory storage (trong production sẽ dùng database)
node_metrics_db: Dict[str, NodeMetrics] = {}


@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "ok",
        "service": "AI Routing Engine",
        "version": "1.0.0"
    }


@app.post("/route/select", response_model=RouteResponse)
async def select_route(request: RouteRequest):
    """
    Chọn route tốt nhất dựa trên AI model

    Args:
        request: RouteRequest với user location và time bucket

    Returns:
        RouteResponse với entry, exit nodes và score
    """
    try:
        # Lấy available nodes
        if request.available_nodes:
            available_nodes = request.available_nodes
        else:
            # Nếu không có, lấy tất cả nodes có metrics
            available_nodes = list(node_metrics_db.keys())

        if len(available_nodes) < 2:
            raise HTTPException(
                status_code=400,
                detail="Cần ít nhất 2 nodes để tạo route"
            )

        # Lấy metrics cho available nodes
        available_metrics = {
            node: node_metrics_db[node]
            for node in available_nodes
            if node in node_metrics_db
        }

        if len(available_metrics) < 2:
            raise HTTPException(
                status_code=400,
                detail="Không đủ node metrics để chọn route"
            )

        # Chọn route bằng bandit
        entry, exit_node, score = bandit.select_arm(
            available_nodes=list(available_metrics.keys()),
            node_metrics=available_metrics,
            time_bucket=request.time_bucket
        )

        return RouteResponse(
            entry=entry,
            exit=exit_node,
            score=round(score, 2)
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/node/metrics")
async def update_node_metrics(metrics: NodeMetrics):
    """
    Cập nhật metrics từ node (từ heartbeat)

    Args:
        metrics: NodeMetrics từ node heartbeat

    Returns:
        Confirmation message
    """
    try:
        # Lưu metrics
        node_metrics_db[metrics.node] = metrics

        # Update bandit
        bandit.update(metrics.node, metrics)

        # Detect anomaly và update reputation
        reputation_update = anomaly_detector.detect_anomaly(metrics)

        return {
            "status": "ok",
            "node": metrics.node,
            "reputation": {
                "score": reputation_update.score,
                "z_score": round(reputation_update.z_score, 3),
                "is_anomaly": reputation_update.is_anomaly
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating metrics: {str(e)}")


@app.get("/node/{node_address}/score")
async def get_node_score(node_address: str, time_bucket: Optional[int] = None):
    """
    Lấy score của một node cụ thể

    Args:
        node_address: Địa chỉ node
        time_bucket: Optional time bucket

    Returns:
        Node score
    """
    if node_address not in node_metrics_db:
        raise HTTPException(status_code=404, detail="Node not found")

    metrics = node_metrics_db[node_address]
    score = bandit.get_node_score(node_address, metrics, time_bucket)

    return {
        "node": node_address,
        "score": round(score, 3),
        "time_bucket": time_bucket
    }


@app.post("/reputation/update", response_model=ReputationUpdate)
async def update_reputation(metrics: NodeMetrics):
    """
    Cập nhật reputation score với anomaly detection

    Args:
        metrics: NodeMetrics

    Returns:
        ReputationUpdate với score và anomaly flag
    """
    try:
        reputation_update = anomaly_detector.detect_anomaly(metrics)
        return reputation_update
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/nodes")
async def list_nodes():
    """
    Liệt kê tất cả nodes có metrics
    """
    nodes = []
    for node_address, metrics in node_metrics_db.items():
        score = bandit.get_node_score(node_address, metrics)
        nodes.append({
            "node": node_address,
            "score": round(score, 3),
            "latency": metrics.latency,
            "loss": metrics.loss,
            "uptime": metrics.uptime
        })

    return {
        "nodes": nodes,
        "count": len(nodes)
    }


@app.post("/url/check", response_model=URLCheckResponse)
async def check_url(request: URLCheckRequest):
    """
    Kiểm tra URL có độc hại không

    Args:
        request: URLCheckRequest với URL cần kiểm tra

    Returns:
        URLCheckResponse với kết quả phân tích
    """
    try:
        result = url_detector.predict(request.url)

        return URLCheckResponse(
            url=request.url,
            is_malicious=result['is_malicious'],
            probability=round(result['probability'], 4),
            confidence=result['confidence']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking URL: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

