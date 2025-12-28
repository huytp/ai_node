"""
Example test script cho AI Routing Engine
"""
import requests
import json
from models import NodeMetrics

BASE_URL = "http://localhost:8000"


def test_route_selection():
    """Test route selection"""
    print("🧪 Testing route selection...")

    # Tạo sample node metrics
    nodes = [
        {"node": "0xNodeA", "latency": 50, "loss": 0.01, "jitter": 5, "uptime": 7200, "bandwidth": 100},
        {"node": "0xNodeB", "latency": 30, "loss": 0.005, "jitter": 3, "uptime": 10800, "bandwidth": 150},
        {"node": "0xNodeC", "latency": 100, "loss": 0.05, "jitter": 15, "uptime": 3600, "bandwidth": 80},
    ]

    # Update metrics
    for node_data in nodes:
        response = requests.post(
            f"{BASE_URL}/node/metrics",
            json=node_data
        )
        print(f"✅ Updated metrics for {node_data['node']}: {response.json()}")

    # Select route
    response = requests.post(
        f"{BASE_URL}/route/select",
        json={
            "time_bucket": 12,
            "available_nodes": ["0xNodeA", "0xNodeB", "0xNodeC"]
        }
    )

    print(f"\n🎯 Selected route: {json.dumps(response.json(), indent=2)}")
    return response.json()


def test_reputation_update():
    """Test reputation update với anomaly detection"""
    print("\n🧪 Testing reputation update...")

    # Normal metrics
    normal_metrics = {
        "node": "0xNodeA",
        "latency": 50,
        "loss": 0.01,
        "jitter": 5,
        "uptime": 7200,
        "bandwidth": 100
    }

    response = requests.post(
        f"{BASE_URL}/reputation/update",
        json=normal_metrics
    )
    print(f"✅ Normal metrics: {json.dumps(response.json(), indent=2)}")

    # Anomaly metrics (high latency)
    anomaly_metrics = {
        "node": "0xNodeA",
        "latency": 2000,  # Very high
        "loss": 0.5,      # Very high
        "jitter": 100,    # Very high
        "uptime": 100,    # Very low
        "bandwidth": 10
    }

    response = requests.post(
        f"{BASE_URL}/reputation/update",
        json=anomaly_metrics
    )
    print(f"⚠️  Anomaly metrics: {json.dumps(response.json(), indent=2)}")


def test_node_score():
    """Test lấy node score"""
    print("\n🧪 Testing node score...")

    response = requests.get(f"{BASE_URL}/node/0xNodeB/score?time_bucket=12")
    print(f"✅ Node score: {json.dumps(response.json(), indent=2)}")


def test_list_nodes():
    """Test list nodes"""
    print("\n🧪 Testing list nodes...")

    response = requests.get(f"{BASE_URL}/nodes")
    print(f"✅ Nodes list: {json.dumps(response.json(), indent=2)}")


if __name__ == "__main__":
    print("🚀 Starting AI Routing Engine tests...\n")

    try:
        # Test health check
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ Health check: {response.json()}\n")

        test_route_selection()
        test_reputation_update()
        test_node_score()
        test_list_nodes()

        print("\n✅ All tests completed!")

    except requests.exceptions.ConnectionError:
        print("❌ Error: Server không chạy. Hãy chạy 'python main.py' trước.")
    except Exception as e:
        print(f"❌ Error: {e}")

