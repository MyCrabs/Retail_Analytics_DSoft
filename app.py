from ultralytics import YOLO
import cv2, os
import numpy as np
import datetime
import csv
from flask import Flask, Response

VIDEO_PATH = "input/cam1_2.mp4"
MODEL_PATH = "weight/yolo11m.pt"
TRACKER_YAML = "BotSort_me.yaml"
ROI_POINTS = np.array([[136,600], [1094,600], [1094,800], [136,800]]) # Cam1
#ROI_POINTS = np.array([[1257,664], [1769,811], [1716,1200], [959,1200]]) # cam2
OUTPUT_DIR = "out/"
CONF_THRESH = 0.5

FACE_MODEL = "weight/yolov8n-face.pt"
FACE_CONF = 0.5
FACE_IMGSZ = 640
FACE_EVERY_N_FRAMES = 3

app = Flask(__name__)

def get_output_name():
    time_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"output_{time_now}.mp4"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, file_name)
    
def get_video_in4(cap):
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return w, h, fps, frame_count

def get_box_id(box):
    if hasattr(box, "id") and box.id is not None:
        try:
            val = box.id 
            if hasattr(val, "__len__"):
                return int(val[0])
            return int(val)
        except Exception:
            try:
                return int(box.id)
            except Exception:
                return -1
    return -1

def expand_head_region(x1,y1,x2,y2,img_w,img_h):
    # đẩy lên trên 15% chiều cao; mở rộng ngang 10% mỗi bên
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return x1,y1,x2,y2
    up = int(0.15*h)
    padx = int(0.1 * w)
    nx1 = max(0,x1-padx)
    ny1 = max(0,y1-up)
    nx2 = min(img_w - 1, x2 + padx)
    ny2 = min(img_h - 1, y2)
    return nx1, ny1, nx2, ny2

def draw_box(frame, results, zone, tracker_data=None, frame_idx=0, fps=30, face_boxes_by_tid=None):
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = int((x1+x2)/2), int(y2)
        inside = cv2.pointPolygonTest(zone, (cx,cy), False) >= 0
        color = (255,255,255) if inside else (255,0,0)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

        tid = get_box_id(box)
        label = f"ID:{tid}"
        # compute displayed times if tracker_data provided
        if tracker_data is not None and tid in tracker_data:
            data = tracker_data[tid]
            total_frames = data.get('total_frames', 0)
            # include ongoing interval
            if data.get('inside') and data.get('enter_frame') is not None:
                ongoing = frame_idx - data['enter_frame']
            else:
                ongoing = 0
            total_secs = (total_frames + ongoing) / max(1.0, fps)
            label = f"ID:{tid} {total_secs:.1f}s"

        # put text above the box
        cv2.putText(frame, label, (x1, max(10, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if face_boxes_by_tid and tid in face_boxes_by_tid:
            fx1,fy1,fx2,fy2 = face_boxes_by_tid[tid]
            cv2.rectangle(frame, (fx1,fy1), (fx2,fy2), (0,255,255), 2)
    return frame
        
def save_tracker_to_csv(tracker_data, fps, output_path):
    csv_path = os.path.splitext(output_path)[0] + ".csv"
    with open(csv_path, mode = "w", newline ="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Track ID", "Time_In_ROI (s)"])
        for tid, data in tracker_data.items():
            secs = data.get("total_frames",0) / max(1.0, fps)
            writer.writerow([tid, f"{secs:.2f}"])

def detect_face(face_model, frame, x1, y1, x2, y2, w, h):
    ex1, ey1, ex2, ey2 = expand_head_region(x1, y1, x2, y2, w, h)
    crop = frame[ey1:ey2, ex1:ex2]
    if crop.size == 0:
        return None
    fres = face_model.predict(crop, conf=FACE_CONF, imgsz=FACE_IMGSZ, verbose=False)[0]
    if len(fres.boxes) == 0:
        return None

    # Chọn mặt lớn nhất
    fboxes = [tuple(map(int, fb.xyxy[0])) for fb in fres.boxes]
    areas = [(x2 - x1)*(y2 - y1) for x1, y1, x2, y2 in fboxes]
    fx1, fy1, fx2, fy2 = fboxes[int(np.argmax(areas))]

    # Đổi về tọa độ gốc frame
    g_fx1 = ex1 + fx1
    g_fy1 = ey1 + fy1
    g_fx2 = ex2 - (crop.shape[1] - fx2)
    g_fy2 = ey2 - (crop.shape[0] - fy2)

    # Giới hạn lại
    g_fx1 = max(0, min(g_fx1, w - 1))
    g_fy1 = max(0, min(g_fy1, h - 1))
    g_fx2 = max(0, min(g_fx2, w - 1))
    g_fy2 = max(0, min(g_fy2, h - 1))
    return (g_fx1, g_fy1, g_fx2, g_fy2)

def update_roi_status(tid, inside, frame_idx, tracker_data):
    if tid not in tracker_data:
        tracker_data[tid] = {'total_frames':0, 'inside':False, 'enter_frame':None}
    data = tracker_data[tid]
    if inside and not data["inside"]:
        data['inside'] = True
        data['enter_frame'] = frame_idx
    elif (not inside) and data["inside"]:
        entered = data.get('enter_frame')
        if entered is not None:
            data['total_frames'] += (frame_idx - entered)
        data['inside'] = False
        data['enter_frame'] = None

def generate_frames():
    model = YOLO(MODEL_PATH)
    face_model = YOLO(FACE_MODEL)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Error: Khong the mo duoc video {VIDEO_PATH}")
    w, h, fps, frame_count = get_video_in4(cap)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    roi_poly = np.array(ROI_POINTS, np.int32).reshape((-1,1,2))
    tracker_data = {}  # tid -> {total_frames:int, inside:bool, enter_frame:int or None}
    frame_idx = 0
    output_path = get_output_name()
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w,h))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(
            frame,
            persist=True,
            tracker = TRACKER_YAML,
            conf = CONF_THRESH,
            classes = [0]
        )
        try:
            res0 = results[0]
        except Exception:
            # fallback: try to get first element
            res_list = list(results)
            res0 = res_list[0] if len(res_list) > 0 else None
        annotated_frame = frame.copy()
        face_boxes_by_tid = {}
        if res0 is not None and hasattr(res0, 'boxes') and res0.boxes is not None:
            for box in res0.boxes:
                tid = get_box_id(box)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = int((x1+x2)/2), int(y2)
                inside = cv2.pointPolygonTest(roi_poly, (cx,cy), False) >= 0
                update_roi_status(tid, inside, frame_idx, tracker_data)
                if inside and (frame_idx % FACE_EVERY_N_FRAMES == 0):
                    face_box = detect_face(face_model, frame, x1,y1,x2,y2,w,h)
                    if face_box: face_boxes_by_tid[tid] = face_box   
        annotated_frame = draw_box(
            annotated_frame, [res0] if res0 is not None else [type('R', (), {'boxes': []})()],
            roi_poly, tracker_data,
            frame_idx, fps,
            face_boxes_by_tid=face_boxes_by_tid)
        #---- Ve vung ROI ----
        cv2.polylines(annotated_frame, [roi_poly], isClosed=True, color = (0,255,0), thickness=2)
        writer.write(annotated_frame)
        ret2, buffer = cv2.imencode(".jpg", annotated_frame)
        frame_bytes = buffer.tobytes()
        yield(b"--frame\r\n"
              b'Content-Type: image/jpeg\r\n\r\n'+frame_bytes+b'\r\n')
        frame_idx += 1
    cap.release()
    writer.release() 
    # finalize tracker_data: Cộng nốt thời gian của những người vẫn ở trong ROI khi video kết thúc
    for tid, data in tracker_data.items():
        if data.get('inside') and data.get('enter_frame') is not None:
            data['total_frames'] += (frame_idx - data['enter_frame'])
            data['inside'] = False
            data['enter_frame'] = None

    save_tracker_to_csv(tracker_data, fps, output_path)
    
@app.route('/')
def index():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port = 1909)
        