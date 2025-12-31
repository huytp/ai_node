# Malicious URL Detection Integration

## Tổng quan

Tính năng phát hiện URL độc hại được tích hợp vào hệ thống deVPN-AI để:
- Phát hiện các URL độc hại trong network traffic
- Gửi push notification chỉ khi phát hiện URL độc hại
- Bảo vệ người dùng khỏi các mối đe dọa trực tuyến

## Kiến trúc

### Backend (AI Routing Service)
- **File**: `ai-routing/url_detector.py`
- **Model**: Logistic Regression với 12 features
- **API Endpoint**: `POST /url/check`

### Mobile App
- **File**: `mobile-app/src/services/networkMonitor.js`
- **Tích hợp**: Gọi API để kiểm tra URL trước khi gửi notification
- **Behavior**: Chỉ gửi notification cho URL độc hại

## Features được sử dụng

1. **URL Length** - Độ dài URL
2. **Number of Dots** - Số dấu chấm
3. **Number of Hyphens** - Số dấu gạch ngang
4. **Subdirectories Count** - Số thư mục con
5. **Parameters Count** - Số tham số query
6. **Has IP Address** - Có địa chỉ IP
7. **Suspicious Keywords** - Từ khóa đáng ngờ
8. **Suspicious TLDs** - Tên miền cấp cao đáng ngờ (.tk, .ml, .ga, etc.)
9. **URL Shortening** - Dịch vụ rút gọn URL
10. **Special Characters** - Ký tự đặc biệt
11. **Digit Ratio** - Tỷ lệ chữ số
12. **Special Char Ratio** - Tỷ lệ ký tự đặc biệt

## Cài đặt

### 1. Cài đặt dependencies

```bash
cd ai-routing
pip install -r requirements.txt
```

### 2. Chạy AI Routing Service

```bash
python main.py
# hoặc
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Cấu hình Mobile App

Thêm vào file `.env` của mobile app:

```env
EXPO_PUBLIC_AI_ROUTING_URL=http://localhost:8000
```

**Lưu ý**:
- Với iOS Simulator/Android Emulator: dùng `http://localhost:8000`
- Với thiết bị thật: dùng IP của máy tính (ví dụ: `http://192.168.1.100:8000`)

## Sử dụng API

### Kiểm tra URL

```bash
curl -X POST "http://localhost:8000/url/check" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/suspicious"}'
```

Response:
```json
{
  "url": "https://example.com/suspicious",
  "is_malicious": true,
  "probability": 0.85,
  "confidence": "high"
}
```

## Testing

Chạy test script:

```bash
cd ai-routing
python test_url_detector.py
```

## Training Data

Model tự động load và train từ `data/urldata.csv` khi khởi động.

File CSV đã có sẵn tại: `ai-routing/data/urldata.csv` (~420k URLs)

### Format CSV

File `data/urldata.csv` có format:
- Header: `url,label`
- Cột 1: `url` - URL cần kiểm tra (có thể không có protocol http://)
- Cột 2: `label` - `good` (benign) hoặc `bad` (malicious)

Model sẽ tự động:
- Phát hiện file CSV tại `data/urldata.csv`
- Thêm `http://` prefix nếu URL thiếu protocol
- Xử lý labels: `bad` → malicious (1), `good` → benign (0)
- Load và train từ CSV
- Lưu model đã train vào `url_detector_model.pkl`

### Dataset Statistics
- Total URLs: ~420,000
- Malicious (`bad`): ~75,536
- Benign (`good`): ~344,821

## Cải thiện Model

Nếu muốn train với dataset khác:

1. Đặt file CSV vào thư mục `ai-routing/` với tên `urldata.csv`
2. Đảm bảo format CSV đúng (xem trên)
3. Xóa file `url_detector_model.pkl` để force retrain
4. Restart service - model sẽ tự động train lại

## Tích hợp với Mobile App

Mobile app tự động:
1. Monitor tất cả HTTP/HTTPS requests
2. Gọi API `/url/check` cho mỗi URL
3. Chỉ gửi push notification nếu `is_malicious = true`

Không cần thay đổi code khác, chỉ cần đảm bảo:
- AI Routing Service đang chạy
- `EXPO_PUBLIC_AI_ROUTING_URL` được cấu hình đúng

## Troubleshooting

### API không phản hồi
- Kiểm tra AI Routing Service có đang chạy không
- Kiểm tra URL trong `.env` có đúng không
- Kiểm tra network connectivity

### Model không load được
- Model sẽ tự động được tạo nếu không tồn tại
- Kiểm tra quyền ghi file trong thư mục `ai-routing`

### False positives/Negatives
- Model hiện tại dùng rule-based synthetic data
- Cần train lại với real dataset để cải thiện accuracy

