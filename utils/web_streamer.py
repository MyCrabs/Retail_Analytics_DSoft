"""
Web streamer - Stream video qua Flask web server
"""
from flask import Flask, Response, render_template_string, request, jsonify
import cv2
import threading
import time

class WebStreamer:
    def __init__(self, port=5000):
        self.port = port
        self.frame = None
        self.lock = threading.Lock()
        self.app = Flask(__name__)
        self.command_queue = []
        self.command_lock = threading.Lock()
        self.setup_routes()
        
    def setup_routes(self): 
        @self.app.route('/')
        def index():
            return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Person Tracking Stream</title>
    <meta charset="UTF-8">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            color: white;
            font-family: 'Segoe UI', Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
        }
        h1 {
            color: #ff601c;
            font-size: 2em;
            text-shadow: 0 0 10px rgba(0, 255, 0, 0.5);
        }
        .video-container {
            background: rgba(0, 0, 0, 0.35);
            padding: 12px;
            border-radius: 12px;
            box-shadow: 0 0 40px rgba(0, 255, 0, 0.25);
            width: 70vw;        /* rộng gần toàn màn hình */
            height: 70vh;       /* cao gần toàn màn hình */
            display: flex;
            align-items: center;
            justify-content: center;
        }
        img {
            width: 100%;
            height: 100%;
            object-fit: contain; /* Giữ đúng tỉ lệ hình, không méo */
            border: 3px solid #00ff00;
            border-radius: 8px;
            display: block;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Person Tracking Stream</h1>
    </div>

    <div class="video-container">
        <img src="{{ url_for('video_feed') }}" id="videoStream" />
    </div>

    <script>
        // Tự động làm mới khung hình mỗi 5 giây để đảm bảo không bị lỗi caching
        setInterval(() => {
            const img = document.getElementById('videoStream');
            img.src = img.src.split('?')[0] + '?' + new Date().getTime();
        }, 5000);
    </script>
</body>
</html>
        ''')

        @self.app.route('/video_feed')
        def video_feed():
            return Response(self._generate(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
    
        @self.app.route('/command', methods= ['POST'])
        def command():
            data = request.get_json()
            cmd = data.get('command', '')
            with self.command_lock:
                self.command_queue.append(cmd)
            return jsonify({'status': 'ok', 'command': cmd})
        
    def _generate(self):
        """Generator cho video stream"""
        while True:
            with self.lock:
                if self.frame is None:
                    continue
                
                # Encode frame thành JPEG
                ret, buffer = cv2.imencode('.jpg', self.frame, 
                                          [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ret:
                    continue
                
                frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.03)  # ~30 FPS
    
    def update_frame(self, frame):
        with self.lock:
            self.frame = frame.copy()
            
    def get_command(self):
        with self.command_lock:
            if len(self.command_queue) > 0:
                return self.command_queue.pop(0)
        return None
    
    def start(self):
        """Start web server trong background thread"""
        def run():
            self.app.run(host='0.0.0.0', port=self.port, 
                        threaded=True, debug=False, use_reloader=False)
        
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        print(f"\n{'='*60}")
        print(f" Web Stream started at:")
        print(f"   http://localhost:{self.port}")
        print(f"   http://0.0.0.0:{self.port}")
        print(f"{'='*60}\n")
        time.sleep(2)  # Đợi server khởi động