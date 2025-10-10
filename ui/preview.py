def show(self, frame, scale=1.0):
        """
        Hiển thị frame
        
        Args:
            frame: ảnh cần hiển thị
            scale: tỷ lệ thu nhỏ (0.5 = 50%)
            
        Returns:
            key code từ waitKey hoặc web command
        """
        # Kiểm tra frame hợp lệ
        if frame is None or frame.size == 0:
            return -1
        
        # Kiểm tra scale hợp lệ
        if scale <= 0:
            scale = 1.0
        
        if self.mode == 'cv2':
            if scale != 1.0 and scale > 0:
                try:
                    preview = cv2.resize(frame, None, fx=scale, fy=scale, 
                                       interpolation=cv2.INTER_LINEAR)
                except cv2.error:
                    preview = frame
            else:
                preview = frame
            
            cv2.imshow(self.window_name, preview)
            return cv2.waitKey(1) & 0xFF
        
        elif self.mode == 'web':
            # Gửi frame lên web stream
            display_frame = frame
            if scale != 1.0 and scale > 0:
                try:
                    display_frame = cv2.resize(frame, None, fx=scale, fy=scale, 
                                     interpolation=cv2.INTER_LINEAR)
                except cv2.error:
                    display_frame = frame
            
            self.web_streamer.update_frame(display_frame)
            
            # Lấy command từ web thay vì stdin
            cmd = self.web_streamer.get_command()
            if cmd:
                # Chuyển command thành key code
                cmd_map = {
                    'start_record': ord('s'),
                    'stop_record': ord('x'),
                    'pause': ord('p'),
                    'quit': ord('q')
                }
                return cmd_map.get(cmd, -1)
            
            return -1
        
        elif self.mode == 'headless':
            # Không hiển thị gì, chỉ đọc input
            return self._read_stdin_key()
import cv2
import sys
import select

class PreviewWindow:
    def __init__(self, window_name="Tracking Preview", width=1280, height=360, 
                 mode='cv2', web_streamer=None):
        """
        Args:
            window_name: tên cửa sổ
            width, height: kích thước cửa sổ
            mode: 'cv2', 'web', hoặc 'headless'
            web_streamer: WebStreamer object nếu mode='web'
        """
        self.window_name = window_name
        self.width = width
        self.height = height
        self.mode = mode
        self.web_streamer = web_streamer
        
        if mode == 'cv2':
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.width, self.height)
        elif mode == 'web':
            if web_streamer is None:
                raise ValueError("web_streamer is required for mode='web'")
        elif mode == 'headless':
            print("🚫 Headless mode - No display")
        
        self.paused = False
        self.quit_requested = False
    
    def show(self, frame, scale=1.0):
        """
        Hiển thị frame
        
        Args:
            frame: ảnh cần hiển thị
            scale: tỷ lệ thu nhỏ (0.5 = 50%)
            
        Returns:
            key code từ waitKey hoặc stdin
        """
        if self.mode == 'cv2':
            if scale != 1.0:
                preview = cv2.resize(frame, None, fx=scale, fy=scale, 
                                   interpolation=cv2.INTER_LINEAR)
            else:
                preview = frame
            
            cv2.imshow(self.window_name, preview)
            return cv2.waitKey(1) & 0xFF
        
        elif self.mode == 'web':
            # Gửi frame lên web stream
            if scale != 1.0:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, 
                                 interpolation=cv2.INTER_LINEAR)
            self.web_streamer.update_frame(frame)
            
            # Đọc input từ stdin (non-blocking)
            return self._read_stdin_key()
    
    def _read_stdin_key(self):
        """Đọc phím từ stdin (cho server mode)"""
        # Non-blocking check nếu có input
        if select.select([sys.stdin], [], [], 0)[0]:
            char = sys.stdin.read(1)
            return ord(char) if char else -1
        return -1
    
    def handle_key(self, key):
        """
        Xử lý phím bấm
        
        Args:
            key: key code từ waitKey
            
        Returns:
            action string: 'quit', 'start_record', 'stop_record', 'pause', None
        """
        if key == -1:
            return None
        
        # q hoặc ESC -> quit
        if key in (ord('q'), 27):
            self.quit_requested = True
            return 'quit'
        
        # s -> start recording
        elif key == ord('s'):
            return 'start_record'
        
        # x -> stop recording
        elif key == ord('x'):
            return 'stop_record'
        
        # p -> pause/resume
        elif key == ord('p'):
            return 'pause'
        
        return None
    
    def pause_loop(self):
        """
        Vòng lặp tạm dừng - chờ 'p' để tiếp tục hoặc 'q' để thoát
        
        Returns:
            True nếu cần quit, False nếu resume
        """
        print("⏸ Tạm dừng. Nhấn 'p' để tiếp tục, 'q'/ESC để thoát.")
        
        if self.mode == 'cv2':
            while True:
                k = cv2.waitKey(50) & 0xFF
                if k in (ord('p'), ord('q'), 27):
                    if k in (ord('q'), 27):
                        self.quit_requested = True
                        return True
                    return False
        else:  # web mode
            import time
            while True:
                k = self._read_stdin_key()
                if k in (ord('p'), ord('q')):
                    if k == ord('q'):
                        self.quit_requested = True
                        return True
                    return False
                time.sleep(0.1)
    
    def destroy(self):
        """Đóng cửa sổ"""
        if self.mode == 'cv2':
            cv2.destroyWindow(self.window_name)
    
    @staticmethod
    def destroy_all():
        """Đóng tất cả cửa sổ OpenCV"""
        cv2.destroyAllWindows()


class KeyboardController:
    """Helper class để print hướng dẫn phím"""
    
    @staticmethod
    def print_help(mode='cv2'):
        """In hướng dẫn sử dụng phím tắt"""
        print("\n" + "="*50)
        print("HƯỚNG DẪN SỬ DỤNG:")
        print("="*50)
        print("  [s] - Bắt đầu ghi video")
        print("  [x] - Dừng ghi video (vẫn live)")
        print("  [p] - Tạm dừng/Tiếp tục")
        print("  [q] hoặc [ESC] - Thoát")
        
        if mode == 'web':
            print("\n📱 WEB MODE:")
            print("  - Mở browser tại URL được hiển thị")
            print("  - Nhập phím trực tiếp trong terminal")
        
        print("="*50 + "\n")