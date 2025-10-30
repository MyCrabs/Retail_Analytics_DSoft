from ultralytics import YOLO
from flask import Flask, Response
import cv2, os, datetime

VIDEO_PATH = "input/cam1.mp4"
FACE_MODEL_PATH = "weights/yolov12n-face.pt"  # model YOLO-face
CONF_THRESH = 0.75                 # ngưỡng confidence
OUTPUT_DIR = "face_full_075"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
def main():
    model = YOLO(FACE_MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Khong the mo duoc video {VIDEO_PATH}")
    frame_idx = 0
    face_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        annotated = frame.copy()
        results = model.predict(frame, conf=CONF_THRESH, verbose=False)
        if len(results[0].boxes) > 0 and len(results)>0:
            for i, box in enumerate(results[0].boxes):
                fx1, fy1, fx2, fy2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (0,255,0),2)
                cv2.putText(annotated, f"{conf:.2f}", (fx1, fy1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                face_crop = frame[fy1:fy2, fx1:fx2]
                if face_crop.size != 0 and conf > 0.6 and frame_idx % 5 == 0:
                    face_name = f"face_{frame_idx:06d}_{face_counter:03d}.jpg"
                    save_path = os.path.join(OUTPUT_DIR, face_name)
                    cv2.imwrite(save_path, face_crop)
                    face_counter += 1
                    
        _, buf = cv2.imencode(".jpg", annotated)
        yield(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+buf.tobytes()+b"\r\n")
    cap.release()
    
@app.route("/")
def index():
    return Response(main(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host = '0.0.0.0', port = 1909, debug = False)