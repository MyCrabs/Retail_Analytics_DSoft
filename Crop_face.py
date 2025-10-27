from ultralytics import YOLO
import cv2, os, numpy as np, datetime
from flask import Flask, Response

# ================= CONFIG =====================
VIDEO_PATH = "input/cam2_2.mp4"
MODEL_PATH = "weight/yolov11n.pt"
FACE_MODEL = "weight/yolov12n-face.pt"
TRACKER_YAML = "BotSort_me.yaml"
ROI_POINTS = np.array([[1257,664], [1769,811], [1716,1200], [959,1200]])  # cam2

CONF_THRESH = 0.5
FACE_CONF = 0.5
FACE_IMGSZ = 640
FACE_MARGIN = 10   # số pixel mở rộng
FACE_DIR = "face/"
OUTPUT_DIR = "out/"

os.makedirs(FACE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)

# ================= HELPERS =====================
def get_video_in4(cap):
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return w, h, fps, total

def get_box_id(box):
    if hasattr(box, "id") and box.id is not None:
        try:
            val = box.id 
            return int(val[0] if hasattr(val, "__len__") else val)
        except Exception:
            return -1
    return -1

def detect_face(face_model, frame, x1, y1, x2, y2, w, h):
    """Phát hiện khuôn mặt trong box người, có thêm margin mở rộng."""
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    fres = face_model.predict(crop, conf=FACE_CONF, imgsz=FACE_IMGSZ, verbose=False)[0]
    if len(fres.boxes) == 0:
        return None

    fboxes = [tuple(map(int, fb.xyxy[0])) for fb in fres.boxes]
    areas = [(x2 - x1)*(y2 - y1) for x1, y1, x2, y2 in fboxes]
    fx1, fy1, fx2, fy2 = fboxes[int(np.argmax(areas))]

    # ====== Thêm margin ======
    fx1 -= FACE_MARGIN
    fy1 -= FACE_MARGIN
    fx2 += FACE_MARGIN
    fy2 += FACE_MARGIN

    # Giới hạn lại trong khung hình
    g_fx1 = max(0, x1 + fx1)
    g_fy1 = max(0, y1 + fy1)
    g_fx2 = min(w - 1, x1 + fx2)
    g_fy2 = min(h - 1, y1 + fy2)
    return (g_fx1, g_fy1, g_fx2, g_fy2)

# ================= STREAMING =====================
def generate_frames():
    model = YOLO(MODEL_PATH)
    face_model = YOLO(FACE_MODEL)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    w, h, fps, total = get_video_in4(cap)
    roi_poly = np.array(ROI_POINTS, np.int32).reshape((-1,1,2))
    frame_idx = 0

    out_path = os.path.join(OUTPUT_DIR, f"stream_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w,h))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.track(frame, persist=True, tracker=TRACKER_YAML, conf=CONF_THRESH, classes=[0])
        if not res or not hasattr(res[0], "boxes"):
            continue

        annotated = frame.copy()
        for box in res[0].boxes:
            tid = get_box_id(box)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (0,255,255)
            cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
            cv2.putText(annotated, f"ID:{tid}", (x1+5, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            # Crop and save face
            face_box = detect_face(face_model, frame, x1, y1, x2, y2, w, h)
            if face_box:
                fx1, fy1, fx2, fy2 = map(int, face_box)
                face_crop = frame[fy1:fy2, fx1:fx2]
                if face_crop.size != 0:
                    fname = os.path.join(FACE_DIR, f"id{tid}_{frame_idx}.jpg")
                    cv2.imwrite(fname, face_crop)

        cv2.polylines(annotated, [roi_poly], True, (0,255,0), 2)
        writer.write(annotated)

        # Encode and stream
        _, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        frame_idx += 1

    cap.release()
    writer.release()

# ================= ROUTE =====================
@app.route('/')
def index():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=1909)
