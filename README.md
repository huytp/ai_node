# AI Routing Engine

## Mục tiêu
- Chọn route tốt nhất
- Không quyết định tiền

## Input Features
- latency
- loss
- jitter
- uptime
- geo distance
- time bucket

## Model (MVP)
- **Contextual Multi-Armed Bandit**
- Reward function:
  ```
  R = -0.5*latency - 0.3*loss + 0.2*stability
  ```

## API

### POST /route/select
Request:
```json
{
  "user_location": "optional",
  "time_bucket": 12,
  "available_nodes": ["0xA", "0xB", "0xC"]
}
```

Response:
```json
{
  "entry": "0xA",
  "exit": "0xB",
  "score": 0.82
}
```

### POST /node/metrics
Cập nhật metrics từ node heartbeat

### GET /node/{address}/score
Lấy score của một node

### POST /reputation/update
Cập nhật reputation với anomaly detection

### GET /nodes
Liệt kê tất cả nodes

## Reputation Update
- Anomaly detection (Z-score)
- Output → backend → blockchain

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Hoặc với uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Documentation

Sau khi chạy server, truy cập:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
