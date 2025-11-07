import cv2
from flask import Flask, Response
import numpy as np
from ultralytics import YOLO
import os

IN = "input/after_lunch_1.mp4"
OUT = "face_padding_roi"
MODEL_PERSON = "weights/yolo11n.pt"
MODEL_FACE = "weights/yolov12n-face.pt"
PADDING = 0.25
PADDING_TOP = 0.4
ROI_P = np.array([(935,463),(1210,509),(1507,1052),(789,1069)], np.int32)

os.makedirs(OUT, exist_ok = True)
cap = cv2.VideoCapture(IN)
person_model = YOLO(MODEL_PERSON)
face_model = YOLO(MODEL_FACE)
app = Flask(__name__)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        overlay = frame.copy()
        cv2.polylines(overlay, [ROI_P], isClosed=True, color = (0,255,255), thickness = 2)
        cv2.fillPoly(overlay, [ROI_P], color = (0,255,255,40))
        alpha = 0.25
        annotated = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        person_result = person_model.track(frame, conf=0.7, persist = True, verbose = False)
        for box in person_result[0].boxes:
            if box.id is None:
                continue
            tid = int(box.id.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = int((x1+x2)/2), int(y2)
            inside = cv2.pointPolygonTest(ROI_P, (cx, cy), False)
            color_person = (0,255,0) if inside >= 0 else (0,0,255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color_person, 2)
            cv2.putText(annotated, f"ID {tid}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_person, 2, cv2.LINE_AA)
            
            if inside < 0:
                continue

            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue
            face_result = face_model.predict(person_crop, conf = 0.75, verbose = False)
            for fbox in face_result[0].boxes:
                fx1, fy1, fx2, fy2 = map(int, fbox.xyxy[0])
                gx1, gy1, gx2, gy2 = x1+fx1, y1+fy1, x1+fx2, y1+fy2
                bw, bh = gx2-gx1, gy2-gy1
                gx1 = max(0, int(gx1 - bw*PADDING))
                gy1 = max(0, int(gy1 - bh*PADDING_TOP))
                gx2 = min(frame.shape[1], int(gx2 + bw*PADDING))
                gy2 = min(frame.shape[0], int(gy2 + bh*PADDING))
                face_crop = frame[gy1:gy2, gx1:gx2]
                if face_crop.size == 0:
                    continue
                cv2.rectangle(annotated, (gx1,gy1), (gx2,gy2), (0,255,255), 2)
                frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if frame_id % 10 == 0:
                    file_name = f"face_id{tid}_{frame_id}.jpg"
                    cv2.imwrite(os.path.join(OUT, file_name), face_crop)
        _, buffer = cv2.imencode(".jpg", annotated)
        frame_bytes = buffer.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
@app.route("/")
def video_feed():
    return Response(generate_frames(), mimetype = 'multipart/x-mixed-replace; boundary=frame')

if __name__== "__main__":
    app.run(host = 'localhost', port = 1909, debug =False)
