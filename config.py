import os
from ultralytics.utils import ROOT

class Config:
    # Đường dẫn
    INPUT_PATH  = "input/cam2.mp4"
    SAVE_DIR = "out/"
    PERSON_WEIGHTS = "weight/yolo11m.pt"
    FACE_WEIGHTS = "weight/yolov8n-face-lindevs.pt"
    HEAD_WEIGHTS = "weight/nano.pt"
    #TRACKER_PATH = str(ROOT / "cfg" / "trackers" / "botsort.yaml")
    TRACKER_PATH = "BotSort_me.yaml"
    
    ENABLE_HEAD_DETECTION = False
    ENABLE_FACE_DETECTION = False
    
    PERSON_CONFIDENCE = 0.5
    HEAD_CONFIDENCE = 0.3
    FACE_CONFIDENCE = 0.3
    
    PERSON_CLASS = [0]
    HEAD_CLASS = [0]
    FACE_CLASS = [0]
    
    
    IMGSZ = 1280
    CLASSES = [0]
    
    #Tracking parameters
    PERSIST = True
    VID_STRIDE = 1
    
    # GPU setting (Bỏ comment nếu có GPU)
    DEVICE = 0
    HALF = True
    
    # UI parameters
    DISPLAY_MODE = "web"
    WEB_PORT = 5000
    PREVIEW_SCALE = 0.5
    SHOW_EVERY = 1
    UPDATE_BAR_EVERY = 20
    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 360
    FRAME_DELAY = 0.00001
    
    # Colors
    COLOR_PERSON = (0, 255, 0)      # Xanh lá - Person bbox
    COLOR_HEAD = (255, 165, 0)      # Cam - Head bbox  
    COLOR_FACE = (255, 0, 255)      # Tím - Face bbox
    COLOR_BBOX = (255,255,255)
    COLOR_RECORDING = (0,0,255)
    COLOR_LIVE = (0,255,0)
    
    @staticmethod
    def ensure_dirs():
        """Tạo thư mục output nếu chưa có"""
        os.makedirs(Config.SAVE_DIR, exist_ok=True)
