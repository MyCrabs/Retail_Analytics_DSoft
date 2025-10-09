"""Xử lí đọc/ghi video"""
import cv2
import os
import datetime

class VideoReader:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Không mở được video input: {video_path}")
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        
        self.cap.release()
        
    def get_properties(self):
        """Tra ve dictionary chua thong tin video"""
        return {
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'total_frames': self.total_frames
        }
    
class VideoWriter:
    def __init__(self, output_dir, width, height, fps, codec='mp4v'):
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.writer = None
        self.output_path = None
        
    def start(self):
        """Bat dau ghi video voi timestamp"""
        if self.writer is not None:
            return False
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"video_{timestamp}.mp4"
        self.output_path = os.path.join(self.output_dir, filename)
        
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        if not self.writer.isOpened():
            self.writer = None
            raise RuntimeError(f"Không tạo được VideoWriter tại: {self.output_path}")
        
        return True
    
    def write(self, frame):
        if self.writer is not None:
            self.writer.write(frame)
            
    def stop(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            return self.output_path
        return None
    
    def is_recording(self):
        return self.writer is not None