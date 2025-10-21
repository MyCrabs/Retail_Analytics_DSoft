from deepface import DeepFace
import cv2

# --- Đường dẫn đến ảnh cần nhận diện ---
img_path = "input/PLy.jpg"   # thay bằng ảnh của bạn

# --- Hiển thị ảnh (tuỳ chọn) ---
img = cv2.imread(img_path)
cv2.imshow("Input Image", img)
cv2.waitKey(500)  # hiển thị 0.5s

# --- Phân tích khuôn mặt ---
result = DeepFace.analyze(
    img_path=img_path,
    actions=['age', 'gender'],
    enforce_detection=False,  # tránh lỗi khi ảnh không rõ mặt
    silent=True
)

# --- In kết quả ---
if isinstance(result, list):
    result = result[0]

print("Predicted Age:", result['age'])
print("Predicted Gender:", result['dominant_gender'])
