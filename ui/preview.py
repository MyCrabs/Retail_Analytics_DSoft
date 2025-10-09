"""Quản lý cửa sổ preview và keyboard control"""
import cv2

class PreviewWindow:
    def __init__(self, window_name = "Tracking Preview", width = 1280, height= 360):
        self.window_name = window_name
        self.width = width
        self.height = height
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)
        self.paused = False
        self.quit_requested = False
        
    def show(self, frame, scale=1.0):
        if scale != 1.0:
            preview = cv2.resize(frame, None, fx = scale, fy=scale,
                                 interpolation = cv2.INTER_LINEAR)
        else:
            preview = frame
            
        cv2.imshow(self.window_name, preview)
        return cv2.waitKey(1) & 0xFF
    
    def handle_key(self, key):
        if key == -1:
            return None
        if key in (ord('q'), 27):  # 'q' or ESC
            self.quit_requested = True
            return 'quit'
        elif key == ord('s'):
            return "start_record"
        elif key == ord('x'):
            return "stop_record"
        elif key == ord('p'):
            return "pause"
        return None
    
    def pause_loop(self):
        print(" || Tam dung. Nhan 'p' de tiep tuc, 'q'/ESC de thoat")
        while True:
            k = cv2.waitKey(50) & 0xFF
            if k in (ord('p'), ord('q'), 27):
                if  k in (ord('q'), 27):
                    self.quit_requested = True
                    return True
                return False
            
    def destroy(self):
        cv2.destroyWindow(self.window_name)
        
    @staticmethod
    def destroy_all():
        cv2.destroyAllWindows()
        
class KeyboardController:
    @staticmethod
    def print_help():
        """In huong dan su dung phim tat"""
        print("\n" + "="*50)
        print("HUONG DAN SU DUNG:")
        print("="*50)
        print(" [s] - Bat dau ghi video")
        print(" [x] - Dung ghi video (van live)")
        print(" [p] - Tam dung/ Tiep tuc")
        print(" [q]/[ESC] - Thoat chuong trinh")
        print("="*50 + "\n")
        