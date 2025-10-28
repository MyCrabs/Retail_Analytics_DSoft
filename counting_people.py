from flask import Flask, Response
import cv2
import numpy as np
import time
from collections import defaultdict
from ultralytics import YOLO

VIDEO_PATH = "input/cam2_2.mp4"
MODEL_PATH = "weights/yolo11m.pt"
TRACKER_PATH = "BotSort_me.yaml"
CONF_THRESH = 0.6

A = (934, 539); B = (55, 878)

app = Flask(__name__)

def line_side(ax, ay, bx, by, px, py):
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)

def draw_dashed_line(img, pt1, pt2, color = (0,255,255), thickness=2, dash_len = 20):
    x1,y1 = pt1
    x2,y2 = pt2
    dist = int(np.hypot(x2-x1, y2-y1)) #Tính độ dài đoạn thẳng AB
    for i in range(0, dist, dash_len*2):
        s = i / dist
        e = min(i+dash_len, dist) / dist
        xs, ys = int(x1+(x2-x1)*s), int(y1+(y2-y1)*s)
        xe, ye = int(x1+(x2-x1)*e), int(y1+(y2-y1)*e)
        cv2.line(img, (xs,ys), (xe,ye), color, thickness, cv2.LINE_AA)

def check_line_cross(tid, cx, cy, last_centroid, last_side, last_cross_frame,
                     frame_idx, v_norm, A, B, MIN_GAP_FRAMES):
    # Xác định vị trí hiện tại so với vạch
    s_now = np.sign(line_side(A[0], A[1], B[0], B[1], cx, cy))
    if s_now ==0:
        s_now = last_side.get(tid, 0)
    
    # Nếu có dữ liệu frame trước đó
    if tid in last_centroid and tid in last_side:
        prev = last_centroid[tid]
        s_prev = last_side[tid]
        # Khi đổi dấu tức là qua vạch
        if s_prev != 0 and s_now != 0 and s_prev != s_now:
            if frame_idx - last_cross_frame[tid] > MIN_GAP_FRAMES:
                delta = np.array([cx-prev[0], cy - prev[1]], dtype=float)
                dir_dot = float(np.dot(delta, v_norm))
                if dir_dot > 0:
                    return True, "(B->A)", (0,255,0)
                else:
                    return True, "(A->B)", (0,0,255)
    # Khong qua vach
    return False, None, None

def generate_frames():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Khong the mo video: {VIDEO_PATH}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    last_centroid, last_side, last_cross_frame = {},{}, defaultdict(lambda: -99999)
    count_in, count_out = 0,0
    frame_idx = 0
    MIN_GAP_FRAMES = int(fps * 2.0)
    
    #Vecto định hướng của vạch
    v_line = np.array([B[0]-A[0], B[1]-A[1]], dtype = float)
    v_norm = v_line / (np.linalg.norm(v_line) + 1e-6)
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        res = model.track(
            frame,
            persist = True,
            conf = CONF_THRESH,
            classes = [0]
            #tracker = TRACKER_PATH
        )
        draw_dashed_line(frame, A, B, color=(0,255,255), thickness=2)
        if res and len(res) >0:
            if res[0].boxes is not None and res[0].boxes.id is not None:
                ids = res[0].boxes.id.int().cpu().tolist()
                xyxy = res[0].boxes.xyxy.cpu().numpy().astype(int)
                
                for i, tid in enumerate(ids):
                    x1,y1,x2,y2 = xyxy[i]
                    cx, cy = (x1+x2) //2, int(y2 - 0.05 * (y2 - y1))
                    
                    cv2.rectangle(frame,(x1,y1), (x2,y2),(255,255,255),2)
                    cv2.putText(frame, f"ID:{tid}", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)
                    crossed, direction, color = check_line_cross(
                        tid, cx, cy, last_centroid, last_side,
                        last_cross_frame, frame_idx, v_norm, A, B, MIN_GAP_FRAMES
                    )
                    if crossed:
                        if "(A->B)" in direction:
                            count_in += 1
                        elif "(B->A)" in direction:
                            count_out += 1
                        last_cross_frame[tid] = frame_idx
                        cv2.putText(frame, direction, (cx+8, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,2)
                        cv2.circle(frame, (cx, cy), 8, color, 3)
                    
                    # Cập nhật thông tin cuối cùng    
                    s_now = np.sign(line_side(A[0], A[1], B[0], B[1], cx, cy))
                    if s_now == 0:
                        s_now = last_side.get(tid, 0)
                    last_centroid[tid] = (cx,cy)
                    last_side[tid] = s_now
        cv2.rectangle(frame, (10,10), (330,100), (0,0,0), -1)
        cv2.putText(frame, f"OUT (B->A): {count_in}", (20,85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        cv2.putText(frame, f"IN (A->B): {count_out}", (20,45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0),2)
        
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        frame_idx += 1
        
@app.route('/')
def index():
    return Response(generate_frames(),
                    mimetype = 'multipart/x-mixed-replace; boundary=frame')
    
if __name__ == "__main__":
    app.run(host = '0.0.0.0', port=1909, debug = False)
