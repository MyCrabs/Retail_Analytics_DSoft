import os
import csv
import cv2
import numpy as np
from tqdm import tqdm
from openvino.runtime import Core  # ⚠️ sửa lại import chính xác

INPUT_DIR = "face_full_075"
OUTPUT_CSV = "face_full_openvino_075_analysis.csv"
MODEL_XML = "weights/age-gender-recognition-retail-0013.xml"
MODEL_BIN = "weights/age-gender-recognition-retail-0013.bin"

ie = Core()
model = ie.read_model(model=MODEL_XML, weights=MODEL_BIN)
compiled_model = ie.compile_model(model=model, device_name="CPU")

input_layer = compiled_model.input(0)
output_layers = compiled_model.outputs

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (62, 62))
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Filename", "Age", "Gender"])
    
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in tqdm(sorted(files), desc=f"Processing {root}", leave=False):
            if not file.lower().endswith(".jpg"):
                continue
            img_path = os.path.join(root, file)
            img = preprocess_image(img_path)
            if img is None:
                writer.writerow([img_path, "", "Error"])
                continue
            try:
                results = compiled_model(img)
                gender_logits = results[output_layers[0]]
                age_output = results[output_layers[1]]

                gender = "Man" if np.argmax(gender_logits) == 1 else "Woman"
                age = int(age_output[0][0] * 100)
                writer.writerow([img_path, age, gender])
            except Exception as e:
                writer.writerow([img_path, "", f"Error: {str(e)}"])
