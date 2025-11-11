import cv2, os
from ultralytics import YOLO
import numpy as np
from flask import Flask, Response

IN = "input/after_lunch_1.mp4"
MODEL_PERSON = "weights/yolo11s.pt"
PADDING = 0.25
PADDING_TOP = 0.4
ROI_P = np.array([(935,463),(1210,509),(1507,1052),(789,1069)], np.int32)
TRACKER = "BotSort_me.yaml"
OUT_VIDEO = "output/tracked_output.mp4"

os.makedirs("output", exist_ok=True)

cap = cv2.VideoCapture(IN)
person_model = YOLO(MODEL_PERSON)
app = Flask(__name__)

fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_writer = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (w, h))

def generate_frames():
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        overlay = frame.copy()
        cv2.polylines(overlay, [ROI_P], isClosed=True, color=(0,255,255), thickness=2)
        cv2.fillPoly(overlay, [ROI_P], color=(0,255,255))
        alpha = 0.25
        annotated = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)

        # Tracking
        person_res = person_model.track(frame, conf=0.75, persist=True,
                                        classes=[0], verbose=False, tracker=TRACKER)

        for box in person_res[0].boxes:
            if box.id is None:
                continue
            tid = int(box.id.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = int((x1+x2)/2), int(y2)
            inside = cv2.pointPolygonTest(ROI_P, (cx, cy), False)
            color_person = (0,255,0) if inside >= 0 else (0,0,255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color_person, 2)
            label = f"ID {tid} | {conf:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_person, 2, cv2.LINE_AA)

        # === Ghi video ===
        out_writer.write(annotated)
        frame_count += 1

        # Stream ra Flask
        _, buffer = cv2.imencode(".jpg", annotated)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Khi video kết thúc
    print(f"Video đã lưu: {OUT_VIDEO}")
    out_writer.release()
    cap.release()

@app.route("/")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1909, debug=False)
