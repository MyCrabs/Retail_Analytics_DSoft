import cv2, dlib, numpy as np, os, matplotlib.pyplot as plt

# === Load models ===
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("weights/shape_predictor_68_face_landmarks.dat")

# === 3D face model points (chuẩn khuôn mặt người trung bình) ===
model_points = np.array([
    (0.0, 0.0, 0.0),            # Mũi
    (0.0, -330.0, -65.0),       # Cằm
    (-225.0, 170.0, -135.0),    # Mắt trái
    (225.0, 170.0, -135.0),     # Mắt phải
    (-150.0, -150.0, -125.0),   # Mép trái
    (150.0, -150.0, -125.0)     # Mép phải
])

def get_head_pose(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        shape = predictor(gray, face)
        image_points = np.array([
            (shape.part(30).x, shape.part(30).y), # Mũi
            (shape.part(8).x, shape.part(8).y),   # Cằm
            (shape.part(36).x, shape.part(36).y), # Mắt trái
            (shape.part(45).x, shape.part(45).y), # Mắt phải
            (shape.part(48).x, shape.part(48).y), # Mép trái
            (shape.part(54).x, shape.part(54).y)  # Mép phải
        ], dtype="double")

        focal_length = image.shape[1]
        center = (image.shape[1]/2, image.shape[0]/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4,1))
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None, None, None

        rmat, _ = cv2.Rodrigues(rotation_vector)
        pose_mat = cv2.hconcat((rmat, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        yaw, pitch, roll = euler_angles.flatten()
        return float(yaw), float(pitch), float(roll)
    return None, None, None

# === Thư mục ảnh cần kiểm tra ===
IMG_DIR = 'face_padding_roi_clahe_restore'

# === Hiển thị từng ảnh kèm góc lệch ===
file_list = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg','.png','.jpeg'))])
n_show = len(file_list)
cols = 6
rows = int(np.ceil(n_show / cols))

plt.figure(figsize=(16, rows * 3))

for idx, fname in enumerate(file_list[:n_show]):
    path = os.path.join(IMG_DIR, fname)
    img = cv2.imread(path)
    if img is None:
        continue

    yaw, pitch, roll = get_head_pose(img)
    if yaw is None:
        title = f"{fname}\nKhông phát hiện mặt"
    else:
        title = f"{fname}\nYaw={yaw:.1f}°, Pitch={pitch:.1f}°, Roll={roll:.1f}°"

    plt.subplot(rows, cols, idx+1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title, fontsize=8)
    plt.axis("off")

plt.tight_layout()
plt.show()
