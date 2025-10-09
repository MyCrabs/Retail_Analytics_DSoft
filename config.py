import os
from ultralytics.utils import ROOT

class Config:
    # Đường dẫn
    INPUT_PATH  = "input/test7.mp4"
    SAVE_DIR = "out/"
    WEIGHTS = "weight/yolov8s.pt"
    #TRACKER_PATH = str(ROOT / "cfg" / "trackers" / "botsort.yaml")
    TRACKER_PATH = "BotSort.yaml"
    # Tham số
    CONFIDENCE = 0.5
    IMGSZ = 960
    CLASSES = [0]
    
    #Tracking parameters
    PERSIST = True
    VID_STRIDE = 1
    
    # GPU setting (Bỏ comment nếu có GPU)
    # DEVICE = 0
    # HALF = True
    
    # UI parameters
    PREVIEW_SCALE = 0.5
    SHOW_EVERY = 1
    UPDATE_BAR_EVERY = 10
    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 360
    
    # Colors
    COLOR_BBOX = (255,255,255)
    COLOR_RECORDING = (0,0,255)
    COLOR_LIVE = (0,255,0)
    
    @staticmethod
    def ensure_dirs():
        """Tạo thư mục output nếu chưa có"""
        os.makedirs(Config.SAVE_DIR, exist_ok=True)
    