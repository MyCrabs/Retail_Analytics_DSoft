"""
Quản lý YOLO model và tracking
"""
from ultralytics import YOLO
import numpy as np

class PersonDetector:
    def __init__(self, weights_path, tracker_config, confidence=0.5, imgsz=960):
        self.model = YOLO(weights_path)
        self.tracker_config = tracker_config
        self.confidence = confidence
        self.imgsz = imgsz
        
    def track(self, source, classes=[0], persist=True, stream=True, 
              vid_stride=1, verbose=False, **kwargs):
        print(f"🔍 Starting tracking with:")
        print(f"   - Source: {source}")
        print(f"   - Confidence: {self.confidence}")
        print(f"   - Image size: {self.imgsz}")
        print(f"   - Classes: {classes}")
        print(f"   - Tracker: {self.tracker_config}")
        
        for res in self.model.track(
            source=source,
            conf=self.confidence,
            imgsz=self.imgsz,
            classes=classes,
            tracker=self.tracker_config,
            persist=persist,
            stream=stream,
            verbose=verbose,
            vid_stride=vid_stride,
            **kwargs
        ):
            yield res
    
    @staticmethod
    def parse_detections(boxes):
        # Debug: in ra boxes
        if boxes is None:
            print("⚠️ boxes is None")
            return np.array([]), np.array([]), np.array([])
        
        if len(boxes) == 0:
            print("⚠️ boxes is empty")
            return np.array([]), np.array([]), np.array([])
        
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        ids = (boxes.id.cpu().numpy().astype(int) 
               if boxes.id is not None 
               else np.array([-1] * len(xyxy)))
        confs = (boxes.conf.cpu().numpy() 
                 if boxes.conf is not None 
                 else np.zeros(len(xyxy)))
        
        return xyxy, ids, confs