from ultralytics import YOLO
import cv2, os, csv, datetime, threading, queue
import numpy as np
from flask import Flask, Response
from deepface import DeepFace

# ===================== CONFIG =====================
VIDEO_PATH = "input/cam2_2.mp4"
MODEL_PATH = "weight/yolo11n.pt"
FACE_MODEL = "weight/yolov8s-face.pt"
TRACKER_YAML = "botsort.yaml"
ROI_POINTS = np.array([[1257,664], [1769,811], [1716,1200], [959,1200]]) # cam2

OUTPUT_DIR = "out/"
CONF_THRESH = 0.5
FACE_CONF = 0.5
FACE_IMGSZ = 640
FACE_QUEUE = queue.Queue(maxsize=20)  # hàng đợi ảnh gửi cho DeepFace

app = Flask(__name__)

# ==================================================
def get_output_name():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    time_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(OUTPUT_DIR, f"output_{time_now}.mp4")

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
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    fres = face_model.predict(crop, conf=FACE_CONF, imgsz=FACE_IMGSZ, verbose=False)[0]
    if len(fres.boxes) == 0:
        return None

    # lấy mặt lớn nhất
    fboxes = [tuple(map(int, fb.xyxy[0])) for fb in fres.boxes]
    areas = [(x2 - x1)*(y2 - y1) for x1, y1, x2, y2 in fboxes]
    fx1, fy1, fx2, fy2 = fboxes[int(np.argmax(areas))]
    g_fx1, g_fy1, g_fx2, g_fy2 = x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2
    return (max(0,g_fx1), max(0,g_fy1), min(w-1,g_fx2), min(h-1,g_fy2))

# ==================================================
# === Thread riêng xử lý DeepFace (không chặn FPS) ===
def deepface_worker(person_info_cache):
    while True:
        try:
            tid, face_crop = FACE_QUEUE.get(timeout=5)
        except queue.Empty:
            continue
        try:
            res = DeepFace.analyze(
                face_crop, actions=['age','gender'],
                enforce_detection=False, silent=True
            )
            if isinstance(res, list): res = res[0]
            age = int(res['age'])
            gender = "Male" if res['dominant_gender'] == "Man" else "Female"

            info = person_info_cache.setdefault(
                tid, {"ages": [], "genders": [], "final_age": None, "final_gender": None}
            )
            if len(info["ages"]) < 10:
                info["ages"].append(age)
                info["genders"].append(gender)
                if len(info["ages"]) == 10:
                    info["final_age"] = int(round(sum(info["ages"]) / len(info["ages"])))
                    info["final_gender"] = max(set(info["genders"]), key=info["genders"].count)
        except Exception as e:
            print("DeepFace Error:", e)

# ==================================================
def update_roi_status(tid, inside, frame_idx, tracker_data, fps):
    if tid not in tracker_data:
        tracker_data[tid] = {
            "inside": False, "enter_frame": None,
            "total_frames": 0, "entry_time": None, "exit_time": None
        }
    data = tracker_data[tid]
    if inside and not data["inside"]:
        data["inside"] = True
        data["enter_frame"] = frame_idx
        data["entry_time"] = datetime.datetime.now().strftime("%H:%M:%S")
    elif (not inside) and data["inside"]:
        if data.get("enter_frame") is not None:
            data["total_frames"] += (frame_idx - data["enter_frame"])
            data["exit_time"] = datetime.datetime.now().strftime("%H:%M:%S")
        data["inside"] = False
        data["enter_frame"] = None

# ==================================================
def save_tracker_to_csv(tracker_data, fps, output_path, person_info_cache):
    csv_path = os.path.splitext(output_path)[0] + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Gender", "Entry Time", "Exit Time", "Dwell Time (s)"])
        for tid, data in tracker_data.items():
            secs = data.get("total_frames", 0) / max(1.0, fps)
            gender = person_info_cache.get(tid, {}).get("final_gender", "")
            writer.writerow([
                tid, gender, 
                data.get("entry_time",""), data.get("exit_time",""),
                f"{secs:.2f}"
            ])
    print(f"CSV saved: {csv_path}")

# ==================================================
def initialize_pipeline():
    model = YOLO(MODEL_PATH)
    face_model = YOLO(FACE_MODEL)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {VIDEO_PATH}")
    w, h, fps, frame_total = get_video_in4(cap)
    output_path = get_output_name()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w,h))
    roi_poly = np.array(ROI_POINTS, np.int32).reshape((-1,1,2))
    return model, face_model, cap, writer, fps, w, h, output_path, roi_poly

def proccess_frame(frame, res, face_model, tracker_data, person_info_cache, roi_poly, fps, frame_idx, w, h):
    annotated = frame.copy()
    for box in res[0].boxes:
        tid = get_box_id(box)
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        cx, cy = int((x1+x2)/2), int(y2)
        inside = cv2.pointPolygonTest(roi_poly, (cx,cy), False) >= 0
        update_roi_status(tid, inside, frame_idx, tracker_data, fps)
        
        # detect face
        face_box = detect_face(face_model, frame, x1, y1, x2, y2, w, h)
        if face_box:
            fx1, fy1, fx2, fy2 = face_box
            face_crop = frame[fy1:fy2, fx1:fx2]
            if face_crop.size != 0 and not FACE_QUEUE.full():
                FACE_QUEUE.put((tid, face_crop))
            
        info = person_info_cache.get(tid, {})
        gender = info.get("final_gender", "")
        age = info.get("final_age", "")
        label = f"ID:{tid}"
        if gender and age:
            label += f" | {gender[:-1]} - {age}"
        elif gender:
            label += f" | {gender[:1]}"
        elif age:
            label += f" | {age}"
        color = (255,180,0) if gender=="Male" else (255,0,255) if gender=="Female" else (200,200,200)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1+5, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.polylines(annotated, [roi_poly], True, (0,255,0), 2)
    return annotated

def main():
    model, face_model, cap, writer, fps, w, h, output_path, roi_poly = initialize_pipeline()
    tracker_data = {}
    person_info_cache = {}
    frame_idx = 0
    # start thread xử lý DeepFace
    threading.Thread(target=deepface_worker, args=(person_info_cache,), daemon=True).start()

    while True:
        ok, frame = cap.read()
        if not ok: break
        results = model.track(frame, persist=True, tracker=TRACKER_YAML, conf=CONF_THRESH, classes=[0])
        if not results or not hasattr(results[0],"boxes"): continue
        annotated = proccess_frame(frame, results, face_model, tracker_data, person_info_cache, roi_poly, fps, frame_idx, w, h)
        writer.write(annotated)
        _, buf = cv2.imencode(".jpg", annotated)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        frame_idx += 1

    cap.release(); writer.release()
    save_tracker_to_csv(tracker_data, fps, output_path, person_info_cache)

# ==================================================
@app.route('/')
def index():
    return Response(main(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=1909)
