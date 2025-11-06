from flask import Flask, render_template_string, Response, request
import cv2

app = Flask(__name__)

VIDEO_PATH = "input/after_lunch_1.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

# ========== TEMPLATE HTML ==========
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Click ROI Points</title>
</head>
<body style="text-align:center; background:#111; color:white;">
    <h2>Chọn ROI bằng cách click vào video</h2>
    <p>Click trên video để lấy tọa độ (hiện ra trong terminal VSCode)</p>
    <img id="video" src="/video_feed" style="border:2px solid white; cursor: crosshair;" />
    <script>
    document.getElementById("video").addEventListener("click", function(e) {
        var rect = this.getBoundingClientRect();
        var x = e.clientX - rect.left;
        var y = e.clientY - rect.top;
        // Gửi toạ độ về Flask
        fetch(`/click?x=${x}&y=${y}`);
    });
    </script>
</body>
</html>
"""

# ========== STREAM VIDEO ==========
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ========== NHẬN TỌA ĐỘ CLICK ==========
@app.route('/click')
def click():
    x = request.args.get('x', type=float)
    y = request.args.get('y', type=float)
    print(f"Clicked at: ({x:.1f}, {y:.1f})")
    return ('', 204)  # Trả về rỗng

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1909, debug=False)
