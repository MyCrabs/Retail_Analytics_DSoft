"""
Retail Analytics - Person Detection, Tracking & Age/Gender Recognition with Flask Streaming
Requirements:
pip install ultralytics opencv-python deepface tf-keras flask
"""

import cv2
from ultralytics import YOLO
from deepface import DeepFace
import numpy as np
from collections import defaultdict
import time
from flask import Flask, render_template_string, Response, jsonify
import threading

app = Flask(__name__)

class RetailAnalytics:
    def __init__(self, video_path, roi_coordinates=None):
        """
        Initialize Retail Analytics system
        
        Args:
            video_path: Path to video file or 0 for webcam
            roi_coordinates: List of (x, y) tuples defining ROI polygon
                           If None, uses entire frame
        """
        self.video_path = video_path
        self.roi_coordinates = roi_coordinates
        
        # Load YOLOv8 model for person detection
        print("Loading YOLO model...")
        self.detection_model = YOLO('yolov8n.pt')
        
        # Storage for tracked persons
        self.tracked_persons = defaultdict(lambda: {
            'age': None,
            'gender': None,
            'confidence': 0,
            'last_analysis': 0,
            'frames_in_roi': 0
        })
        
        # Analysis interval (seconds)
        self.analysis_interval = 2.0
        
        # Statistics
        self.stats = {
            'total_visitors': 0,
            'male': 0,
            'female': 0,
            'age_groups': defaultdict(int)
        }
        
        # Video capture
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.lock = threading.Lock()
        
    def is_in_roi(self, center_point):
        """Check if point is inside ROI polygon"""
        if self.roi_coordinates is None:
            return True
        
        x, y = center_point
        roi_array = np.array(self.roi_coordinates, dtype=np.int32)
        result = cv2.pointPolygonTest(roi_array, (x, y), False)
        return result >= 0
    
    def get_age_group(self, age):
        """Categorize age into groups"""
        if age < 18:
            return "0-17"
        elif age < 25:
            return "18-24"
        elif age < 35:
            return "25-34"
        elif age < 45:
            return "35-44"
        elif age < 55:
            return "45-54"
        else:
            return "55+"
    
    def analyze_face(self, frame, bbox):
        """Analyze age and gender from face crop"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Add padding to bbox
            h, w = frame.shape[:2]
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            face_crop = frame[y1:y2, x1:x2]
            
            # Skip if crop is too small
            if face_crop.shape[0] < 50 or face_crop.shape[1] < 50:
                return None, None
            
            # Analyze with DeepFace
            result = DeepFace.analyze(
                face_crop, 
                actions=['age', 'gender'],
                enforce_detection=False,
                silent=True
            )
            
            # Handle both single result and list of results
            if isinstance(result, list):
                result = result[0]
            
            age = int(result['age'])
            gender = result['dominant_gender']
            
            return age, gender
            
        except Exception as e:
            return None, None
    
    def update_statistics(self, track_id, age, gender):
        """Update visitor statistics"""
        person = self.tracked_persons[track_id]
        
        # Only count if this is new information
        if person['age'] is None:
            person['age'] = age
            person['gender'] = gender
            
            # Update stats
            self.stats['total_visitors'] += 1
            
            if gender == 'Man':
                self.stats['male'] += 1
            else:
                self.stats['female'] += 1
            
            age_group = self.get_age_group(age)
            self.stats['age_groups'][age_group] += 1
    
    def draw_roi(self, frame):
        """Draw ROI polygon on frame"""
        if self.roi_coordinates is not None:
            roi_array = np.array(self.roi_coordinates, dtype=np.int32)
            cv2.polylines(frame, [roi_array], True, (0, 255, 255), 2)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [roi_array], (0, 255, 255))
            cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
    
    def draw_info(self, frame, bbox, track_id, in_roi):
        """Draw bounding box and information"""
        x1, y1, x2, y2 = map(int, bbox)
        person = self.tracked_persons[track_id]
        
        # Color based on ROI status
        color = (0, 255, 0) if in_roi else (128, 128, 128)
        
        # Draw bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Prepare text
        texts = [f"ID: {track_id}"]
        if person['age'] is not None:
            texts.append(f"{person['gender']}, {person['age']}y")
        
        # Draw text background and text
        y_offset = y1 - 10
        for text in texts:
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y_offset - text_h - 5), 
                         (x1 + text_w + 5, y_offset), color, -1)
            cv2.putText(frame, text, (x1 + 3, y_offset - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            y_offset -= (text_h + 10)
    
    def draw_statistics(self, frame):
        """Draw statistics panel"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent panel
        panel_h = 200
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw statistics text
        y_pos = 35
        line_height = 25
        
        cv2.putText(frame, "RETAIL ANALYTICS", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += line_height + 5
        
        cv2.putText(frame, f"Total Visitors: {self.stats['total_visitors']}", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += line_height
        
        cv2.putText(frame, f"Male: {self.stats['male']} | Female: {self.stats['female']}", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += line_height
        
        # Age groups
        cv2.putText(frame, "Age Groups:", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += line_height
        
        for age_group in sorted(self.stats['age_groups'].keys()):
            count = self.stats['age_groups'][age_group]
            cv2.putText(frame, f"  {age_group}: {count}", (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_pos += line_height - 5
    
    def process_frame(self, frame):
        """Process a single frame"""
        current_time = time.time()
        
        # Run YOLO tracking
        results = self.detection_model.track(
            frame, 
            persist=True, 
            classes=[0],  # 0 = person
            verbose=False
        )
        
        # Process detections
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for bbox, track_id in zip(boxes, track_ids):
                # Get center point
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                
                # Check if in ROI
                in_roi = self.is_in_roi((center_x, center_y))
                
                person = self.tracked_persons[track_id]
                
                if in_roi:
                    person['frames_in_roi'] += 1
                    
                    # Analyze age/gender if needed
                    time_since_analysis = current_time - person['last_analysis']
                    
                    if (person['age'] is None or 
                        time_since_analysis > self.analysis_interval):
                        
                        age, gender = self.analyze_face(frame, bbox)
                        
                        if age is not None and gender is not None:
                            self.update_statistics(track_id, age, gender)
                            person['last_analysis'] = current_time
                
                # Draw information
                self.draw_info(frame, bbox, track_id, in_roi)
        
        # Draw ROI
        self.draw_roi(frame)
        
        # Draw statistics
        self.draw_statistics(frame)
        
        return frame
    
    def start_capture(self):
        """Start video capture"""
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        # Get video properties
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {width}x{height} @ {fps}fps")
        self.is_running = True
    
    def generate_frames(self):
        """Generate frames for streaming"""
        frame_count = 0
        
        while self.is_running:
            ret, frame = self.cap.read()
            
            if not ret:
                # Loop video
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            frame_count += 1
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Store current frame
            with self.lock:
                self.current_frame = processed_frame.copy()
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame, 
                                      [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Progress log
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames | Visitors: {self.stats['total_visitors']}")
    
    def stop_capture(self):
        """Stop video capture"""
        self.is_running = False
        if self.cap:
            self.cap.release()
    
    def get_statistics(self):
        """Get current statistics as dict"""
        return {
            'total_visitors': self.stats['total_visitors'],
            'male': self.stats['male'],
            'female': self.stats['female'],
            'age_groups': dict(self.stats['age_groups'])
        }


# Global analyzer instance
analyzer = None

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Retail Analytics Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .video-container {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        
        .video-container h2 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.5em;
        }
        
        .video-wrapper {
            position: relative;
            width: 100%;
            padding-bottom: 56.25%; /* 16:9 aspect ratio */
            background: #000;
            border-radius: 10px;
            overflow: hidden;
        }
        
        .video-wrapper img {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        
        .stats-container {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        
        .stats-container h2 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.5em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .stat-card h3 {
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 10px;
        }
        
        .stat-card .value {
            font-size: 2.5em;
            font-weight: bold;
        }
        
        .gender-stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 15px;
        }
        
        .gender-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        
        .gender-card.male {
            border-left: 4px solid #4285f4;
        }
        
        .gender-card.female {
            border-left: 4px solid #ea4335;
        }
        
        .gender-card h4 {
            color: #666;
            font-size: 0.9em;
            margin-bottom: 5px;
        }
        
        .gender-card .count {
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
        }
        
        .age-groups {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
        }
        
        .age-groups h3 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.1em;
        }
        
        .age-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #e0e0e0;
        }
        
        .age-item:last-child {
            border-bottom: none;
        }
        
        .age-label {
            font-weight: 500;
            color: #555;
        }
        
        .age-count {
            background: #667eea;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }
        
        .status-bar {
            background: white;
            border-radius: 15px;
            padding: 15px 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .status-dot {
            width: 12px;
            height: 12px;
            background: #4caf50;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .refresh-btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.3s;
        }
        
        .refresh-btn:hover {
            background: #764ba2;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        
        @media (max-width: 1024px) {
            .main-content {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛒 Retail Analytics Dashboard</h1>
            <p>Real-time Customer Analysis & Tracking</p>
        </div>
        
        <div class="main-content">
            <div class="video-container">
                <h2>📹 Live Camera Feed</h2>
                <div class="video-wrapper">
                    <img src="{{ url_for('video_feed') }}" alt="Video Stream">
                </div>
            </div>
            
            <div class="stats-container">
                <h2>📊 Analytics</h2>
                
                <div class="stat-card">
                    <h3>Total Visitors</h3>
                    <div class="value" id="total-visitors">0</div>
                </div>
                
                <div class="gender-stats">
                    <div class="gender-card male">
                        <h4>👨 Male</h4>
                        <div class="count" id="male-count">0</div>
                    </div>
                    <div class="gender-card female">
                        <h4>👩 Female</h4>
                        <div class="count" id="female-count">0</div>
                    </div>
                </div>
                
                <div class="age-groups">
                    <h3>Age Distribution</h3>
                    <div id="age-groups-list">
                        <!-- Age groups will be populated here -->
                    </div>
                </div>
            </div>
        </div>
        
        <div class="status-bar">
            <div class="status-indicator">
                <div class="status-dot"></div>
                <span><strong>System Active</strong> - Processing live feed</span>
            </div>
            <button class="refresh-btn" onclick="updateStats()">🔄 Refresh Stats</button>
        </div>
    </div>
    
    <script>
        function updateStats() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('total-visitors').textContent = data.total_visitors;
                    document.getElementById('male-count').textContent = data.male;
                    document.getElementById('female-count').textContent = data.female;
                    
                    // Update age groups
                    const ageGroupsList = document.getElementById('age-groups-list');
                    ageGroupsList.innerHTML = '';
                    
                    const ageOrder = ['0-17', '18-24', '25-34', '35-44', '45-54', '55+'];
                    ageOrder.forEach(group => {
                        const count = data.age_groups[group] || 0;
                        const ageItem = document.createElement('div');
                        ageItem.className = 'age-item';
                        ageItem.innerHTML = `
                            <span class="age-label">${group} years</span>
                            <span class="age-count">${count}</span>
                        `;
                        ageGroupsList.appendChild(ageItem);
                    });
                })
                .catch(error => console.error('Error fetching stats:', error));
        }
        
        // Auto-refresh stats every 3 seconds
        setInterval(updateStats, 3000);
        
        // Initial load
        updateStats();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(analyzer.generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    """Statistics API endpoint"""
    return jsonify(analyzer.get_statistics())

def run_flask_app(host='0.0.0.0', port=1909):
    """Run Flask app"""
    print(f"\n{'='*60}")
    print(f"🚀 Retail Analytics Dashboard Starting...")
    print(f"{'='*60}")
    print(f"📱 Access the dashboard at: http://localhost:{port}")
    print(f"🌐 Network access: http://{host}:{port}")
    print(f"{'='*60}\n")
    
    app.run(host=host, port=port, debug=False, threaded=True)

# Main execution
if __name__ == "__main__":
    # Define ROI (Region of Interest)
    roi = [
        (1010,234),
        (1846,704),
        (1566,1058),
        (502,820)
    ]
    
    # Initialize analyzer
    analyzer = RetailAnalytics(
        video_path="input/test7.mp4",
        roi_coordinates=roi  # or None for full frame
    )
    
    # Start video capture
    analyzer.start_capture()
    
    # Start Flask app
    try:
        run_flask_app(host='0.0.0.0', port=1909)
    except KeyboardInterrupt:
        print("\n🛑 Stopping analytics system...")
        analyzer.stop_capture()
        print("✅ System stopped successfully")