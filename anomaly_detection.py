"""
Anomaly Detection sử dụng Z-score cho reputation update
"""
import numpy as np
from scipy import stats
from typing import List, Dict
from models import NodeMetrics, ReputationUpdate


class AnomalyDetector:
    """
    Phát hiện anomaly trong node metrics sử dụng Z-score
    """

    def __init__(self, threshold: float = 2.5):
        """
        Args:
            threshold: Z-score threshold (default 2.5 = 99% confidence)
        """
        self.threshold = threshold
        self.metrics_history: Dict[str, List[float]] = {
            'latency': [],
            'loss': [],
            'jitter': [],
            'uptime': []
        }

    def update_history(self, metrics: NodeMetrics):
        """Cập nhật lịch sử metrics"""
        self.metrics_history['latency'].append(metrics.latency)
        self.metrics_history['loss'].append(metrics.loss)
        self.metrics_history['jitter'].append(metrics.jitter)
        self.metrics_history['uptime'].append(metrics.uptime)

        # Giữ lịch sử tối đa 1000 samples
        for key in self.metrics_history:
            if len(self.metrics_history[key]) > 1000:
                self.metrics_history[key] = self.metrics_history[key][-1000:]

    def calculate_z_score(self, value: float, metric_type: str) -> float:
        """
        Tính Z-score cho một metric value

        Args:
            value: Giá trị hiện tại
            metric_type: 'latency', 'loss', 'jitter', 'uptime'

        Returns:
            Z-score
        """
        history = self.metrics_history[metric_type]

        if len(history) < 10:  # Cần ít nhất 10 samples
            return 0.0

        mean = np.mean(history)
        std = np.std(history)

        if std == 0:
            return 0.0

        z_score = (value - mean) / std
        return z_score

    def detect_anomaly(self, metrics: NodeMetrics) -> ReputationUpdate:
        """
        Phát hiện anomaly và tính reputation score

        Returns:
            ReputationUpdate với score và anomaly flag
        """
        # Tính Z-score cho từng metric
        latency_z = abs(self.calculate_z_score(metrics.latency, 'latency'))
        loss_z = abs(self.calculate_z_score(metrics.loss, 'loss'))
        jitter_z = abs(self.calculate_z_score(metrics.jitter, 'jitter'))
        uptime_z = abs(self.calculate_z_score(metrics.uptime, 'uptime'))

        # Z-score tổng hợp (weighted average)
        # Latency và loss quan trọng hơn
        combined_z = (latency_z * 0.3 + loss_z * 0.3 + jitter_z * 0.2 + uptime_z * 0.2)

        # Phát hiện anomaly
        is_anomaly = combined_z > self.threshold

        # Tính reputation score [0-100]
        # Score giảm nếu có anomaly
        if is_anomaly:
            # Score giảm dựa trên mức độ anomaly
            score = max(0, int(100 - (combined_z - self.threshold) * 20))
        else:
            # Score dựa trên chất lượng metrics
            latency_score = max(0, 100 - int(metrics.latency / 10))  # 0-100ms = 100, 1000ms = 0
            loss_score = max(0, 100 - int(metrics.loss * 1000))  # 0% = 100, 10% = 0
            jitter_score = max(0, 100 - int(metrics.jitter / 2))  # 0ms = 100, 200ms = 0
            uptime_score = min(100, int(metrics.uptime / 360))  # 1h = 10, 10h = 100

            score = int((latency_score * 0.3 + loss_score * 0.3 +
                        jitter_score * 0.2 + uptime_score * 0.2))

        # Cập nhật lịch sử
        self.update_history(metrics)

        return ReputationUpdate(
            node=metrics.node,
            score=score,
            z_score=combined_z,
            is_anomaly=is_anomaly
        )

    def get_reputation_score(self, metrics: NodeMetrics) -> int:
        """
        Lấy reputation score mà không cần detect anomaly
        """
        update = self.detect_anomaly(metrics)
        return update.score

