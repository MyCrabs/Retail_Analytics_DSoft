from ultralytics import YOLO
import cv2

VIDEO_PATH = "input/after_lunch.mp4"     
MODEL_PATH = "weights/yolov12n-face.pt"     
OUTPUT_PATH = "output_face.mp4"    

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Không thể mở video {VIDEO_PATH}")

fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Dự đoán khuôn mặt
    results = model.predict(frame, conf=0.7, verbose=False)

    # Vẽ box
    annotated_frame = results[0].plot()

    # Ghi ra file và hiển thị
    #out.write(annotated_frame)
    cv2.imshow("Face Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ======= DỌN DẸP =======
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video đã lưu tại: {OUTPUT_PATH}")
