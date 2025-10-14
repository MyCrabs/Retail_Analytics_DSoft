import time
import cv2
from tqdm import tqdm
from models.detector import MultiModelDetector
from utils.video_io import VideoReader, VideoWriter
from utils.drawing import DrawingUtils
from utils.web_streamer import WebStreamer
from ui.preview import PreviewWindow

class PersonTracker:
    def __init__(self, config):
        """
        Args:
            config: Config object
        """
        self.config = config
        
        # Khởi tạo multi-model detector
        self.detector = MultiModelDetector(config)
        
        # Đọc thông tin video
        self.video_reader = VideoReader(config.INPUT_PATH)
        self.video_props = self.video_reader.get_properties()
        
        # Khởi tạo writer (chưa start)
        self.writer = VideoWriter(
            output_dir=config.SAVE_DIR,
            width=self.video_props['width'],
            height=self.video_props['height'],
            fps=self.video_props['fps']
        )
        
        # Khởi tạo web streamer nếu cần
        web_streamer = None
        if config.DISPLAY_MODE == 'web':
            web_streamer = WebStreamer(port=config.WEB_PORT)
            web_streamer.start()
        
        # UI
        self.preview = PreviewWindow(
            width=min(config.WINDOW_WIDTH, self.video_props['width']),
            height=max(config.WINDOW_HEIGHT, 
                      int(min(config.WINDOW_WIDTH, self.video_props['width']) 
                          * self.video_props['height'] / max(self.video_props['width'], 1))),
            mode=config.DISPLAY_MODE,
            web_streamer=web_streamer
        )
        
        # Metrics
        self.frame_idx = 0
        self.ema_fps = Noneweb_streamer = WebStreamer(port=config.WEB_PORT)
        web_streamer.start()
        
        # UI
        self.preview = PreviewWindow(
            width=min(config.WINDOW_WIDTH, self.video_props['width']),
            height=max(config.WINDOW_HEIGHT, 
                      int(min(config.WINDOW_WIDTH, self.video_props['width']) 
                          * self.video_props['height'] / max(self.video_props['width'], 1))),
            mode=config.DISPLAY_MODE,
            web_streamer=web_streamer
        )
        
        # Metrics
        self.frame_idx = 0
        self.ema_fps = None
        
    def run(self):
        """Chạy tracking chính với cascade detection"""
        # Debug config
        print(f"🔧 Config:")
        print(f"   PREVIEW_SCALE: {self.config.PREVIEW_SCALE}")
        print(f"   DISPLAY_MODE: {self.config.DISPLAY_MODE}")
        print(f"   SHOW_EVERY: {self.config.SHOW_EVERY}")
        print(f"   HEAD Detection: {'✅' if self.config.ENABLE_HEAD_DETECTION else '❌'}")
        print(f"   FACE Detection: {'✅' if self.config.ENABLE_FACE_DETECTION else '❌'}")
        
        total_frames = self.video_props['total_frames']
        pbar_total = total_frames if total_frames and total_frames > 0 else None
        
        pbar = tqdm(total=pbar_total, unit="frame", desc="Tracking", ncols=90)
        t0 = time.perf_counter()
        
        try:
            # Tracking loop
            tracking_kwargs = {}
            if hasattr(self.config, 'DEVICE'):
                tracking_kwargs['device'] = self.config.DEVICE
            if hasattr(self.config, 'HALF'):
                tracking_kwargs['half'] = self.config.HALF
            
            for res in self.detector.track_persons(
                source=self.config.INPUT_PATH,
                **tracking_kwargs
            ):
                t1 = time.perf_counter()
                self.frame_idx += 1
                
                # Lấy frame gốc
                frame = res.orig_img
                
                # Debug frame
                if frame is None:
                    print(f"⚠️ Frame {self.frame_idx} is None!")
                    continue
                if frame.size == 0:
                    print(f"⚠️ Frame {self.frame_idx} is empty!")
                    continue
                
                # Parse person detections
                person_xyxy, person_ids, person_confs = MultiModelDetector.parse_detections(res.boxes)
                
                # Counters
                total_heads = 0
                total_faces = 0
                all_heads = []
                all_faces = []
                
                # Cascade detection: Person → Head → Face
                if len(person_xyxy) > 0:
                    # Vẽ person boxes trước
                    frame = DrawingUtils.draw_person_detections(
                        frame, person_xyxy, person_ids, person_confs,
                        color=self.config.COLOR_PERSON
                    )
                    
                    # Detect heads trong mỗi person
                    if self.config.ENABLE_HEAD_DETECTION:
                        for person_bbox in person_xyxy:
                            heads = self.detector.detect_heads_in_person(frame, person_bbox)
                            all_heads.extend(heads)
                            total_heads += len(heads)
                            
                            # Detect faces trong mỗi head
                            if self.config.ENABLE_FACE_DETECTION:
                                for head_bbox in heads:
                                    faces = self.detector.detect_faces_in_head(frame, head_bbox)
                                    all_faces.extend(faces)
                                    total_faces += len(faces)
                    
                    # Vẽ head boxes
                    if all_heads:
                        frame = DrawingUtils.draw_head_boxes(
                            frame, all_heads,
                            color=self.config.COLOR_HEAD
                        )
                    
                    # Vẽ face boxes
                    if all_faces:
                        frame = DrawingUtils.draw_face_boxes(
                            frame, all_faces,
                            color=self.config.COLOR_FACE
                        )
                
                # Tính FPS (EMA)
                dt_proc = max(time.perf_counter() - t1, 1e-6)
                inst_fps = 1.0 / dt_proc
                self.ema_fps = (inst_fps if self.ema_fps is None 
                               else 0.9 * self.ema_fps + 0.1 * inst_fps)
                
                # Vẽ HUD với stats
                stats = {
                    'persons': len(person_xyxy),
                    'heads': total_heads,
                    'faces': total_faces
                }
                frame = DrawingUtils.draw_hud(
                    frame,
                    is_recording=self.writer.is_recording(),
                    fps=self.ema_fps,
                    frame_idx=self.frame_idx,
                    stats=stats,
                    color_rec=self.config.COLOR_RECORDING,
                    color_live=self.config.COLOR_LIVE
                )
                
                # Ghi video nếu đang record
                if self.writer.is_recording():
                    self.writer.write(frame)
                
                # Hiển thị preview
                key = -1
                if self.frame_idx % self.config.SHOW_EVERY == 0:
                    key = self.preview.show(frame, scale=self.config.PREVIEW_SCALE)
                else:
                    # Không hiển thị nhưng vẫn đọc phím
                    # Đối với web mode, vẫn cần update frame
                    if self.config.DISPLAY_MODE == 'web':
                        key = self.preview.show(frame, scale=self.config.PREVIEW_SCALE)
                    else:
                        key = cv2.waitKey(1) & 0xFF
                
                # Xử lý phím
                action = self.preview.handle_key(key)
                if action == 'quit':
                    break
                elif action == 'start_record':
                    self._start_recording()
                elif action == 'stop_record':
                    self._stop_recording()
                elif action == 'pause':
                    if self.preview.pause_loop():
                        break
                
                # Delay để điều khiển tốc độ hiển thị
                if hasattr(self.config, 'FRAME_DELAY') and self.config.FRAME_DELAY > 0:
                    time.sleep(self.config.FRAME_DELAY)
                
                # Update progress bar (ít thường xuyên hơn)
                status = "REC" if self.writer.is_recording() else "LIVE"
                if pbar_total:
                    if self.frame_idx % self.config.UPDATE_BAR_EVERY == 0:
                        pbar.update(self.config.UPDATE_BAR_EVERY)
                        pbar.set_postfix({
                            "frame": self.frame_idx, 
                            "fps": f"{self.ema_fps:.1f}",
                            "P": len(person_xyxy),
                            "H": total_heads,
                            "F": total_faces,
                            "save": status
                        })
                else:
                    if self.frame_idx % self.config.UPDATE_BAR_EVERY == 0:
                        pbar.update(self.config.UPDATE_BAR_EVERY)
                        pbar.set_postfix({
                            "frame": self.frame_idx, 
                            "fps": f"{self.ema_fps:.1f}",
                            "P": len(person_xyxy),
                            "H": total_heads,
                            "F": total_faces,
                            "save": status
                        })
        
        except KeyboardInterrupt:
            print("⏹ Dừng sớm (Ctrl+C).")
        
        finally:
            self._cleanup(pbar, t0)
    
    def _start_recording(self):
        """Bắt đầu ghi video"""
        if not self.writer.is_recording():
            try:
                self.writer.start()
                print(f"⏺ Bắt đầu ghi → {self.writer.output_path}")
            except RuntimeError as e:
                print(f"❌ Lỗi: {e}")
    
    def _stop_recording(self):
        """Dừng ghi video"""
        output_path = self.writer.stop()
        if output_path:
            print(f"⏹ Dừng ghi → {output_path}")
    
    def _cleanup(self, pbar, start_time):
        """Dọn dẹp resources"""
        output_path = self.writer.stop()
        if output_path:
            print(f"✅ Đã lưu: {output_path}")
        else:
            print("ℹ️ Không lưu file (LIVE only).")
        
        pbar.close()
        dt_all = time.perf_counter() - start_time
        avg_fps = self.frame_idx / max(dt_all, 1e-6)
        print(f"Frames: {self.frame_idx} | Time: {dt_all:.2f}s | ~FPS: {avg_fps:.1f}")
        
        self.preview.destroy_all()