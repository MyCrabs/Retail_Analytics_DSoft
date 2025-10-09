# Retail Analytics – Person/Head/Face Detection & Age/Gender Classification
## 1. Overview 
Hệ thống Retail Analytics cho phép nhận diện và phân tích khách hàng trong môi trường bán lẻ (siêu thị, cửa hàng, trung tâm thương mại).
Chức năng chính:
Person detection & tracking: phát hiện và theo dõi chuyển động của khách hàng.
Head & face detection: xử lý trường hợp đông người hoặc bị che khuất.
Age & gender classification (tùy chọn): thống kê nhân khẩu học.
ROI analytics: xác định khu vực quan tâm (lối vào, quầy thanh toán, v.v).

## 2. Project Structure
```
project/
├─ main.py                 # Entry point chính
├─ config.py               # Cấu hình tập trung (đường dẫn, ngưỡng, model)
│
├─ models/
│  ├─ __init__.py
│  └─ detector.py          # Quản lý YOLO model (person/head/face)
│
├─ utils/
│  ├─ __init__.py
│  ├─ video_io.py          # Đọc/ghi video, xử lý input/output
│  ├─ drawing.py           # Vẽ bbox, ROI, FPS, HUD
│  └─ roi_utils.py         # Xử lý vùng ROI, polygon check
│
├─ tracking/
│  ├─ __init__.py
│  └─ tracker.py           # Logic tracking (BoT-SORT, ByteTrack)
│
└─ ui/
   ├─ __init__.py
   └─ preview.py           # Quản lý hiển thị, phím tắt, record video
```