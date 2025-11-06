from ultralytics import YOLO
from flask import Flask, Response
from deepface import DeepFace
import cv2, os, numpy as np

VIDEO_PATH = "input/after_lunch_1.mp4"
MODEL_PERSON = "weights/yolo11n.pt"
MODEL_FACE = "weights/yolov12n-face.pt"
TRACKER_YAML = "botsort.yaml"   # file cấu hình tracker YOLO
FACE_DIR = "face_crop_directory_padding"

os.makedirs(FACE_DIR, exist_ok=True)

app = Flask(__name__)
person_model = YOLO(MODEL_PERSON)
face_model = YOLO(MODEL_FACE)
cap = cv2.VideoCapture(VIDEO_PATH)
PADDING = 0.25
PADDING_TOP = 0.4

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        person_results = person_model.track(
            frame, conf=0.7, persist=True, verbose=False
        )
        annotated = frame.copy()
        for box in person_results[0].boxes:
            if box.id is None:
                continue
            tid = int(box.id.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            face_results = face_model.predict(person_crop, conf=0.75, verbose=False)
            for fbox in face_results[0].boxes:
                fx1, fy1, fx2, fy2 = map(int, fbox.xyxy[0])
                # Chuyển tọa độ face về khung gốc frame
                gx1, gy1, gx2, gy2 = x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2

                bw, bh = gx2 - gx1, gy2 - gy1
                gx1 = max(0, int(gx1 - bw * PADDING))
                gy1 = max(0, int(gy1 - bh * PADDING_TOP))
                gx2 = min(frame.shape[1], int(gx2 + bw * PADDING))
                gy2 = min(frame.shape[0], int(gy2 + bh * PADDING))
                face_crop = frame[gy1:gy2, gx1:gx2]
                if face_crop.size == 0:
                    continue
                try:
                    pred = DeepFace.analyze(face_crop, actions=["gender"], enforce_detection=False)
                    gender = pred[0]["dominant_gender"]
                    color = (255, 204, 153) if gender == "Man" else (255, 153, 255)
                except Exception:
                    color = (200, 200, 200)
                    gender = "Unknown"
                cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), color, 2)
                cv2.putText(
                    annotated, f"ID {tid} | {gender}",
                    (gx1, gy1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
                )
                frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if frame_id % 10 == 0:
                    file_name = f"face_id{tid}_{frame_id}.jpg"
                    cv2.imwrite(os.path.join(FACE_DIR, file_name), face_crop)
        _, buffer = cv2.imencode(".jpg", annotated)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route("/")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="localhost", port=1909, debug=False)
