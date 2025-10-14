from ultralytics import YOLO
import numpy as np
import cv2

class MultiModelDetector:
    def __init__(self, config):
        """
        Args:
            config: Config object chứa thông tin models
        """
        self.config = config
        
        # Load Person model (bắt buộc)
        print(f"Loading Person model: {config.PERSON_WEIGHTS}")
        self.person_model = YOLO(config.PERSON_WEIGHTS)
        
        # Load Head model (optional)
        self.head_model = None
        if config.ENABLE_HEAD_DETECTION:
            try:
                print(f"Loading Head model: {config.HEAD_WEIGHTS}")
                self.head_model = YOLO(config.HEAD_WEIGHTS)
            except Exception as e:
                print(f"⚠️ Failed to load Head model: {e}")
                self.head_model = None
        
        # Load Face model (optional)
        self.face_model = None
        if config.ENABLE_FACE_DETECTION:
            try:
                print(f"Loading Face model: {config.FACE_WEIGHTS}")
                self.face_model = YOLO(config.FACE_WEIGHTS)
            except Exception as e:
                print(f"⚠️ Failed to load Face model: {e}")
                self.face_model = None
        
        self.tracker_config = config.TRACKER_PATH
        self.imgsz = config.IMGSZ
    
    def track_persons(self, source, **kwargs):
        """
        Track persons trong video
        
        Yields:
            results object từ YOLO
        """
        for res in self.person_model.predict(
            source=source,
            conf=self.config.PERSON_CONFIDENCE,
            imgsz=self.imgsz,
            classes=self.config.PERSON_CLASS,
            stream=True,
            verbose=False,
            vid_stride=self.config.VID_STRIDE,
            **kwargs
        ):
            yield res
    
    def detect_heads_in_person(self, frame, person_bbox):
        """
        Detect heads trong person bbox
        
        Args:
            frame: full frame
            person_bbox: (x1, y1, x2, y2) của person
            
        Returns:
            list of head bboxes [(x1, y1, x2, y2), ...]
        """
        if self.head_model is None:
            return []
        
        x1, y1, x2, y2 = person_bbox
        
        # Crop person region
        person_crop = frame[y1:y2, x1:x2]
        
        if person_crop.size == 0:
            return []
        
        # Detect heads trong crop
        results = self.head_model.predict(
            person_crop,
            conf=self.config.HEAD_CONFIDENCE,
            classes=self.config.HEAD_CLASS,
            verbose=False
        )
        
        heads = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            for box in boxes:
                # Convert local coords → global coords
                hx1, hy1, hx2, hy2 = box
                heads.append((x1 + hx1, y1 + hy1, x1 + hx2, y1 + hy2))
        
        return heads
    
    def detect_faces_in_head(self, frame, head_bbox):
        """
        Detect faces trong head bbox
        
        Args:
            frame: full frame
            head_bbox: (x1, y1, x2, y2) của head
            
        Returns:
            list of face bboxes [(x1, y1, x2, y2), ...]
        """
        if self.face_model is None:
            return []
        
        x1, y1, x2, y2 = head_bbox
        
        # Crop head region
        head_crop = frame[y1:y2, x1:x2]
        
        if head_crop.size == 0:
            return []
        
        # Detect faces trong crop
        results = self.face_model.predict(
            head_crop,
            conf=self.config.FACE_CONFIDENCE,
            classes=self.config.FACE_CLASS,
            verbose=False
        )
        
        faces = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            for box in boxes:
                # Convert local coords → global coords
                fx1, fy1, fx2, fy2 = box
                faces.append((x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2))
        
        return faces
    
    @staticmethod
    def parse_detections(boxes):
        if boxes is None or len(boxes) == 0:
            return np.array([]), np.array([]), np.array([])
        
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        ids = (boxes.id.cpu().numpy().astype(int) 
               if boxes.id is not None 
               else np.array([-1] * len(xyxy)))
        confs = (boxes.conf.cpu().numpy() 
                 if boxes.conf is not None 
                 else np.zeros(len(xyxy)))
        
        return xyxy, ids, confs