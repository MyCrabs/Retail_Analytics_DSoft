from deepface import DeepFace
import os
import csv
from tqdm import tqdm

INPUT_DIR = "faces_out"
OUTPUT_CSV = "faces_out.csv"

with open(OUTPUT_CSV, mode = "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Filename","Age","Gender"])
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in tqdm(sorted(files), desc = f"Proccessing {root}", leave= False):
            if not file.lower().endswith(".jpg"):
                continue
            img_path = os.path.join(root, file)
            try:
                res = DeepFace.analyze(
                    img_path,
                    actions = ["age","gender"],
                    enforce_detection=False,
                    silent= True
                )
                if isinstance(res, list):
                    res = res[0]
                age= int(res.get("age", -1))
                gender = res.get("dominant_gender","Unknown")
                writer.writerow([img_path, age, gender])
            except Exception as e:
                writer.writerow([img_path,"","Error"])
