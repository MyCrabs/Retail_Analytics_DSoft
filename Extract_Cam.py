from flask import Flask, render_template_string, Response, request
import cv2, threading, datetime

app = Flask(__name__)

RTSP_URL = "rtsp://admin:Dsoft%402024@192.168.2.211:554"

cap = cv2.VideoCapture(RTSP_URL)
recording = False
writer = None
lock = threading.Lock()

# ====== HTML TEMPLATE ======
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <title>Camera Stream</title>
  <style>
    body { background: #111; color: white; text-align: center; font-family: Arial; }
    video { border: 3px solid #4CAF50; margin-top: 20px; }
    button { 
      margin: 10px; padding: 10px 20px; font-size: 16px; 
      border: none; border-radius: 5px; cursor: pointer;
    }
    .record { background-color: #e74c3c; color: white; }
    .stop { background-color: #555; color: white; }
  </style>
</head>
<body>
  <h1>Camera Stream</h1>
  <img src="{{ url_for('video_feed') }}" width="70%" />
  <div>
    <button class="record" onclick="fetch('/start_record')">Start Record</button>
    <button class="stop" onclick="fetch('/stop_record')">Stop Record</button>
  </div>
</body>
</html>
"""

# ====== STREAM GENERATOR ======
def generate_frames():
    global cap, recording, writer
    while True:
        success, frame = cap.read()
        if not success:
            continue
        # Ghi video nếu đang record
        with lock:
            if recording and writer is not None:
                writer.write(frame)
        # Encode frame sang JPEG để stream ra trình duyệt
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ====== ROUTES ======
@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_record')
def start_record():
    global writer, recording
    with lock:
        if not recording:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"record_{ts}.mp4"
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
            recording = True
            print(f"Bắt đầu ghi video: {filename}")
            return f"Recording started: {filename}"
        else:
            return "Đã ghi rồi!"

@app.route('/stop_record')
def stop_record():
    global writer, recording
    with lock:
        if recording:
            recording = False
            if writer is not None:
                writer.release()
                writer = None
            print("Dừng ghi video.")
            return "Recording stopped"
        else:
            return "Chưa có tiến trình ghi nào đang chạy."

# ====== MAIN ======
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1909, debug=False, threaded=True)
