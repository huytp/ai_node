"""
Contextual Multi-Armed Bandit implementation cho route selection
"""
import numpy as np
from typing import List, Dict, Optional
from models import NodeMetrics


class ContextualBandit:
    """
    Contextual Multi-Armed Bandit cho route selection

    Reward function: R = -0.5*latency - 0.3*loss + 0.2*stability
    """

    def __init__(self, alpha: float = 0.1, epsilon: float = 0.1):
        """
        Args:
            alpha: Learning rate
            epsilon: Exploration rate
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.node_features: Dict[str, np.ndarray] = {}
        self.node_rewards: Dict[str, List[float]] = {}
        self.node_weights: Dict[str, np.ndarray] = {}

    def _extract_features(self, metrics: NodeMetrics, time_bucket: Optional[int] = None) -> np.ndarray:
        """
        Extract features từ node metrics

        Features: [latency, loss, jitter, uptime_norm, geo_distance_norm, time_bucket]
        """
        # Normalize features
        latency_norm = min(metrics.latency / 1000.0, 1.0)  # Normalize to [0, 1], max 1000ms
        loss_norm = metrics.loss  # Already [0, 1]
        jitter_norm = min(metrics.jitter / 100.0, 1.0)  # Normalize to [0, 1], max 100ms
        uptime_norm = min(metrics.uptime / 86400.0, 1.0)  # Normalize to [0, 1], max 24h
        geo_norm = min((metrics.geo_distance or 0) / 20000.0, 1.0) if metrics.geo_distance else 0.5  # Max 20000km
        time_norm = (time_bucket or 12) / 24.0  # Normalize hour to [0, 1]

        return np.array([latency_norm, loss_norm, jitter_norm, uptime_norm, geo_norm, time_norm])

    def _calculate_reward(self, metrics: NodeMetrics) -> float:
        """
        Tính reward dựa trên metrics
        R = -0.5*latency - 0.3*loss + 0.2*stability
        """
        # Normalize latency (ms) - lower is better
        latency_score = 1.0 - min(metrics.latency / 500.0, 1.0)  # Max 500ms

        # Loss - lower is better
        loss_score = 1.0 - metrics.loss

        # Stability = f(uptime, jitter) - higher uptime, lower jitter = better
        uptime_score = min(metrics.uptime / 3600.0, 1.0)  # Normalize to 1h
        jitter_score = 1.0 - min(metrics.jitter / 50.0, 1.0)  # Max 50ms jitter
        stability = (uptime_score + jitter_score) / 2.0

        # Reward function
        reward = -0.5 * (1.0 - latency_score) - 0.3 * (1.0 - loss_score) + 0.2 * stability

        # Normalize to [0, 1]
        reward = (reward + 1.0) / 2.0

        return max(0.0, min(1.0, reward))

    def update(self, node: str, metrics: NodeMetrics, reward: Optional[float] = None):
        """
        Update bandit với metrics mới

        Args:
            node: Node address
            metrics: Node metrics
            reward: Optional reward (nếu không có sẽ tính từ metrics)
        """
        if reward is None:
            reward = self._calculate_reward(metrics)

        features = self._extract_features(metrics)
        self.node_features[node] = features

        if node not in self.node_rewards:
            self.node_rewards[node] = []
            self.node_weights[node] = np.zeros(len(features))

        self.node_rewards[node].append(reward)

        # Update weights using gradient descent
        predicted = np.dot(self.node_weights[node], features)
        error = reward - predicted
        self.node_weights[node] += self.alpha * error * features

    def select_arm(self, available_nodes: List[str], node_metrics: Dict[str, NodeMetrics],
                   time_bucket: Optional[int] = None) -> tuple:
        """
        Chọn best route (entry, exit) từ available nodes

        Returns:
            (entry_node, exit_node, score)
        """
        if len(available_nodes) < 2:
            raise ValueError("Cần ít nhất 2 nodes để tạo route")

        scores = {}

        for node in available_nodes:
            if node not in node_metrics:
                continue

            metrics = node_metrics[node]
            features = self._extract_features(metrics, time_bucket)

            # Predict reward
            if node in self.node_weights:
                predicted_reward = np.dot(self.node_weights[node], features)
            else:
                # Initial reward estimate
                predicted_reward = self._calculate_reward(metrics)

            # Epsilon-greedy: exploration
            if np.random.random() < self.epsilon:
                predicted_reward = np.random.random()

            scores[node] = max(0.0, min(1.0, predicted_reward))

        if not scores:
            raise ValueError("Không có node metrics available")

        # Sort nodes by score
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Select best entry and exit (different nodes)
        entry = sorted_nodes[0][0]
        exit_node = sorted_nodes[1][0] if len(sorted_nodes) > 1 else sorted_nodes[0][0]

        # Route score = average of entry and exit
        route_score = (scores[entry] + scores[exit_node]) / 2.0

        return entry, exit_node, route_score

    def get_node_score(self, node: str, metrics: NodeMetrics,
                      time_bucket: Optional[int] = None) -> float:
        """Lấy score của một node"""
        features = self._extract_features(metrics, time_bucket)

        if node in self.node_weights:
            score = np.dot(self.node_weights[node], features)
        else:
            score = self._calculate_reward(metrics)

        return max(0.0, min(1.0, score))

