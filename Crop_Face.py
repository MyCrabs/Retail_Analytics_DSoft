from ultralytics import YOLO
from flask import Flask, Response
import cv2, os, datetime

VIDEO_PATH = "input/cam1_2.mp4"
FACE_MODEL_PATH = "weights/yolov12n-face.pt"  # model YOLO-face
PERSON_MODEL = "weights/yolo11n.pt"    # model YOLO-person
CONF_THRESH = 0.8                  # ngưỡng confidence
OUTPUT_DIR = "faces_out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
def main():
    model = YOLO(FACE_MODEL_PATH)
    person_model = YOLO(PERSON_MODEL)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Khong the mo duoc video {VIDEO_PATH}")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        person_result = person_model.track(
            frame, persist=True, conf=CONF_THRESH, classes=[0]
        )
        annotated = frame.copy()
        if person_result and len(person_result[0].boxes) > 0:
            for box in person_result[0].boxes:
                tid = int(box.id[0]) if box.id is not None else -1
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                person_crop = frame[y1:y2, x1:x2]
                result_face = model.predict(person_crop, conf=CONF_THRESH, verbose=False)
                if len(result_face[0].boxes) > 0:
                    fx1, fy1, fx2, fy2 = map(int, result_face[0].boxes[0].xyxy[0])
                    face_crop = person_crop[fy1:fy2, fx1:fx2]
                    conf = float(result_face[0].boxes[0].conf[0])
                    global_fx1, global_fy1 = x1 + fx1, y1 + fy1
                    global_fx2, global_fy2 = x1 + fx2, y1 + fy2
                    cv2.rectangle(annotated, (global_fx1,global_fy1), (global_fx2,global_fy2), (0,255,0),2)
                    cv2.putText(annotated, f"{conf:.2f}", (global_fx1, global_fy1-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    if face_crop.size != 0 and conf > 0.6 and frame_idx % 5 == 0:
                        id_folder = os.path.join(OUTPUT_DIR, f"id_{tid:03d}")
                        os.makedirs(id_folder, exist_ok=True)
                        face_name = f"frame_{frame_idx:06d}.jpg"
                        cv2.imwrite(os.path.join(id_folder, face_name), face_crop)
                    
        _, buf = cv2.imencode(".jpg", annotated)
        yield(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+buf.tobytes()+b"\r\n")
    cap.release()
    
@app.route("/")
def index():
    return Response(main(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host = '0.0.0.0', port = 1909, debug = False)