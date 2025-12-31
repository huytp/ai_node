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

### POST /url/check
Kiểm tra URL có độc hại không

Request:
```json
{
  "url": "https://example.com/suspicious-page"
}
```

Response:
```json
{
  "url": "https://example.com/suspicious-page",
  "is_malicious": true,
  "probability": 0.85,
  "confidence": "high"
}
```

## Malicious URL Detection
- Sử dụng Logistic Regression để phát hiện URL độc hại
- Feature extraction từ URL characteristics:
  - URL length, số dots, hyphens
  - Số subdirectories, parameters
  - Có IP address, suspicious keywords, suspicious TLDs
  - URL shortening services
  - Special characters ratio
- Model được train từ `urldata.csv` (nếu có) hoặc synthetic data (fallback)
- Model được lưu tại `url_detector_model.pkl`

### Training Data
Model tự động load và train từ `data/urldata.csv` khi khởi động.

File CSV đã có sẵn tại: `ai-routing/data/urldata.csv`

Format CSV:
- Cột 1: `url` - URL cần kiểm tra (có thể không có protocol)
- Cột 2: `label` - `good` (benign) hoặc `bad` (malicious)

Model sẽ tự động:
- Thêm `http://` prefix nếu URL thiếu protocol
- Xử lý labels: `bad` → malicious (1), `good` → benign (0)
- Train với toàn bộ dataset (~420k URLs)

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
