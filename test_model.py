from openvino import Core
import cv2
import numpy as np

# === 1. Load mô hình OpenVINO ===
core = Core()
model_path = "weight/age-gender-recognition-retail-0013.xml"  # Đường dẫn tới model
compiled_model = core.compile_model(model_path, "CPU")

# Lấy output layer (0 = age, 1 = gender)
age_output = compiled_model.output(0)
gender_output = compiled_model.output(1)

# === 2. Đọc ảnh khuôn mặt bất kỳ ===
face_path = "input/PLy.jpg"  # <-- thay bằng đường dẫn ảnh thật
face = cv2.imread(face_path)
if face is None:
    raise FileNotFoundError(f"Không tìm thấy ảnh: {face_path}")

# === 3. Tiền xử lý ảnh ===
img = cv2.resize(face, (62, 62))
img = img.transpose((2, 0, 1))[None, :]  # HWC -> NCHW, thêm batch
img = img.astype(np.float32) / 255.0

# === 4. Dự đoán bằng mô hình ===
res = compiled_model([img])

# === 5. Hậu xử lý output ===
age = float(res[age_output][0][0][0][0]) * 100
prob = float(res[gender_output][0][0][0][0])
gender = "Female" if prob > 0.5 else "Male"
confidence = abs(prob - 0.5) * 2

# === 6. In kết quả ===
print(f"Predicted Age: {int(round(age))} years")
print(f"Predicted Gender: {gender} (conf: {confidence:.2f})")

# === 7. Hiển thị ảnh (tùy chọn) ===
cv2.putText(face, f"{gender}, {int(round(age))}y", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
cv2.imshow("Age-Gender Prediction", face)
cv2.waitKey(0)
cv2.destroyAllWindows()
