import cv2

class DrawingUtils:
    @staticmethod
    def draw_person_detections(img, xyxy, ids, confs, color=(0, 255, 0)):
        for (x1, y1, x2, y2), tid, cf in zip(xyxy, ids, confs):
            # Vẽ bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Label với background
            label = f"Person ID{tid}" if tid != -1 else "Person"
            
            # Tính kích thước text
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Vẽ background cho text
            label_y = max(th + 4, y1 - 4)
            cv2.rectangle(img, (x1, label_y - th - 4), (x1 + tw + 4, label_y), color, -1)
            
            # Vẽ text
            cv2.putText(img, label, (x1 + 2, label_y - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return img
    
    @staticmethod
    def draw_head_boxes(img, head_boxes, color=(255, 165, 0)):
        for (x1, y1, x2, y2) in head_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Label nhỏ hơn
            cv2.putText(img, "Head", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return img
    
    @staticmethod
    def draw_face_boxes(img, face_boxes, color=(255, 0, 255)):
        for (x1, y1, x2, y2) in face_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Label nhỏ hơn
            cv2.putText(img, "Face", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        
        return img
    
    @staticmethod
    def draw_hud(img, is_recording, fps, frame_idx=None, stats=None,
                color_rec=(0, 0, 255), color_live=(0, 255, 0)):
        status = "REC \u25CF" if is_recording else "LIVE"
        status_color = color_rec if is_recording else color_live
        
        y_offset = 28
        
        cv2.putText(img, status, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
        
        y_offset += 30
        cv2.putText(img, f"FPS:{fps:.1f}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if frame_idx is not None:
            y_offset += 28
            cv2.putText(img, f"Frame:{frame_idx}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if stats:
            y_offset += 25
            cv2.putText(img, f"Persons:{stats.get('persons', 0)}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            y_offset += 20
            cv2.putText(img, f"Heads:{stats.get('heads', 0)}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
            
            y_offset += 20
            cv2.putText(img, f"Faces:{stats.get('faces', 0)}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        return img