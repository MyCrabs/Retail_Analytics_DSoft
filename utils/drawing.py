"""
Các hàm vẽ UI: bounding box, HUD
"""
import cv2

class DrawingUtils:
    @staticmethod
    def draw_detections(img, xyxy, ids, confs, color=(0, 255, 0), debug=False):
        img_h, img_w = img.shape[:2]
        
        for i, ((x1, y1, x2, y2), tid, cf) in enumerate(zip(xyxy, ids, confs)):
            # Debug: In ra tọa độ (chỉ khi cần)
            if debug:
                print(f"  Box {i+1}: [{x1},{y1},{x2},{y2}] ID={tid} conf={cf:.2f}")
            
            if x2 <= x1 or y2 <= y1:
                if debug:
                    print(f"    ⚠️ Invalid box size!")
                continue
            
            if x1 >= img_w or y1 >= img_h or x2 <= 0 or y2 <= 0:
                if debug:
                    print(f"    ⚠️ Box outside frame! (frame size: {img_w}x{img_h})")
                continue
            
            # Clip bbox vào trong frame
            x1 = max(0, min(x1, img_w-1))
            y1 = max(0, min(y1, img_h-1))
            x2 = max(0, min(x2, img_w-1))
            y2 = max(0, min(y2, img_h-1))
            
            # Vẽ bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Label với background
            label = f"ID {tid} {cf:.2f}" if tid != -1 else f"Person {cf:.2f}"
            
            # Tính kích thước text
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            
            # Vẽ background cho text
            label_y = max(th + 4, y1 - 4)
            cv2.rectangle(img, (x1, label_y - th - 4), (x1 + tw + 4, label_y), color, -1)
            
            # Vẽ text
            cv2.putText(img, label, (x1 + 2, label_y - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, 
                       lineType=cv2.LINE_8)
            
            if debug:
                print(f"    ✅ Drew box at [{x1},{y1},{x2},{y2}]")
        
        return img
    
    @staticmethod
    def draw_hud(img, is_recording, fps, frame_idx=None,
                color_rec=(0, 0, 255), color_live=(0, 255, 0)):
        status = "REC \u25CF" if is_recording else "LIVE"
        status_color = color_rec if is_recording else color_live
        
        cv2.putText(img, status, (10, 28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2, 
                   lineType=cv2.LINE_8)
        
        cv2.putText(img, f"FPS:{fps:.1f}", (10, 58),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                   lineType=cv2.LINE_8)
        
        if frame_idx is not None:
            cv2.putText(img, f"Frame:{frame_idx}", (10, 88),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1,
                       lineType=cv2.LINE_8)
        
        return img