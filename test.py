import cv2, os

FACE_DIR = "face/"

for file in os.listdir(FACE_DIR):
    path = os.path.join(FACE_DIR, file)
    if os.path.isfile(path):
        img = cv2.imread(path)
        if img is None:
            print(f"{file}: Không đọc được ảnh")
            continue
        h, w, _ = img.shape
        print(f"{file}: {w}x{h}")
