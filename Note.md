# Thứ 2 (27/10): Đếm người qua vạch "(Counting_people.py)
## 1. Mục tiêu
- Task này sẽ đếm số lượng người đi qua vạch (cổng)
- Xác định hướng di chuyển IN hoặc OUT.
## 2. Phương pháp
- Sử dụng BotSORT để tracking -> trả về ID + bounding box
- Dùng numpy để tính toán vecto và hướng di chuyển
- Sử dụng trung điểm cạnh dưới của bbox để làm mốc của 1 người:
x = (x1+x2)/2 , y = y2
- Sử dụng MIN_GAP_FRAMES để xử lý việc nhập nhằng tại vạch, giảm đc vài trường hợp sai sót của IN
## 2.1 Kiểm tra vị trí so với vạch (Cross Product)
- cross = (xB - xA)(yP-yA) - (yB-yA)(xP-xA)
- Nếu cross > 0 -> điểm P nằm bên trái hướng A->B, ngược lại
- Nếu cross = 0: điểm P nằm trên đường AB
## 2.2 Kiểm tra hướng di chuyển (Dot Product)
- Sau khi biết đã qua vạch, ta cần xác định hướng đi nào (IN/OUT)
- delta = (x_now - x_prev, y_now - y_prev): là vecto di chuyển của người
- v_norm = (xB-xA, yB-yA) là hướng của vạch A-B
- dot = delta . v_norm: Nếu dot > 0 thì người di chuyển cùng hướng với vạch -> IN và ngược lại

# Thứ 3 (28/10)
## 1. Mục tiêu
- Thực hiện bài toán xác định và giới tính thông qua YOLO-face & DeepFace
## 2. Phương pháp
- Sử dụng model YOLO12-face để detect khuôn mặt từ frame (THRESH_HOLD = 0.8)
- Lưu theo face_counter_xxx/frame_xxx.jpg
- Sau đó dùng DeepFace.analyze để xác định giới tính và tuổi. Kết quả được lưu vào file CSV 3 cột Filename, Age, Gender
### 3. Nhận xét và kết quả thử nghiệm
- Khác với bài toán đếm người, chỉ cần phát hiện được người đi qua vạch, bài toán này yêu cầu khuôn mặt rõ nét hơn => Tăng ngưỡng confidence

CONF	Số lượng ảnh crop được	                  Nhận xét
0.6	          444 ảnh	           Nhiều ảnh nhưng có khuôn mặt mờ, khó nhận dạng
0.7	          374 ảnh	           Giảm bớt nhiễu, khuôn mặt rõ hơn
0.8	          225 ảnh	           Ít ảnh hơn nhưng chất lượng tốt nhất, DeepFace phân tích ổn định hơn

- Với ngưỡng 0.8 kết quả Male: 74%, Female: 26%
=> Kết quả cải thiện rõ rệt so với tuần trc
### 4. Hướng nghiên cứu và cải tiến
- Kết hợp tracking ID từ YOLO-person
+ Mỗi người đc gán id riêng nhờ mô hình tracking
+ Dựa trên bbox của person, có thể crop lại vùng khuôn mặt bên trong để gắn face_id tương ứng
- Ưu điểm: Mỗi người có thể được theo dõi xuyên suốt video -> Gom được nhiều khuôn mặt cùng ID
- Hạn chế: Một số bbox của person ở xa camera vẫn được tracker lưu lại -> Các face được crop ra từ bbox này nhỏ và mờ (Ảnh chất lượng thấp)
=> Vậy cần đánh đổi số lượng và chất lượng khuôn mặt