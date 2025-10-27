# Ngày 2: Đếm người qua vạch "(Counting_people.py)
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
### 