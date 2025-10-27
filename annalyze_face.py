import os
import pandas as pd
from deepface import DeepFace
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import torch

# ============ CONFIG =============
FACE_DIR = "face/"
OUT_CSV = "face_result.csv"
DETECTOR_BACKEND = "retinaface"   # RetinaFace chạy bằng PyTorch
MAX_WORKERS = 6                   # GPU mạnh → có thể tăng lên 6–8
BATCH_SIZE = 500                  # Ghi CSV theo đợt để tránh mất dữ liệu

os.makedirs(FACE_DIR, exist_ok=True)

# ============ CHECK GPU ==========
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"🚀 PyTorch GPU detected: {gpu_name}")
else:
    print("⚠️ Không phát hiện GPU — DeepFace sẽ chạy bằng CPU (chậm hơn nhiều).")

# ============ LOAD MODELS ============
print("⏳ Loading models into memory...")
models = DeepFace.build_model('DeepFace')  # Dùng model chung cho age+gender
print("✅ Models loaded and ready (using PyTorch backend).")

# ============ ANALYSIS FUNCTION ============
def analyze_image(path):
    """Phân tích một ảnh khuôn mặt."""
    try:
        result = DeepFace.analyze(
            img_path=path,
            actions=['age', 'gender'],
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            #models=models   # ✅ Reuse model đã load
        )[0]

        return {
            "filename": os.path.basename(path),
            "age": result.get("age"),
            "gender": result.get("gender"),
            "dominant_gender": result.get("dominant_gender", "")
        }

    except Exception as e:
        return {"filename": os.path.basename(path), "error": str(e)}

# ============ MAIN PIPELINE ============
def analyze_faces(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder)
             if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not files:
        print("⚠️ Không có ảnh hợp lệ trong thư mục face/")
        return

    print(f"🖼️ Tổng số ảnh cần phân tích: {len(files)}")
    results = []
    completed = 0

    # ThreadPool cho phép xử lý song song
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(analyze_image, f): f for f in files}

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="🚀 Đang phân tích khuôn mặt"):
            res = future.result()
            results.append(res)
            completed += 1

            # Ghi CSV định kỳ (tránh mất dữ liệu khi chạy lâu)
            if completed % BATCH_SIZE == 0:
                pd.DataFrame(results).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
                print(f" Đã lưu tạm {completed} ảnh vào {OUT_CSV}")

    # Ghi file kết quả cuối
    pd.DataFrame(results).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n Hoàn tất! Tổng {len(results)} ảnh, kết quả lưu tại: {OUT_CSV}")


if __name__ == "__main__":
    analyze_faces(FACE_DIR)
