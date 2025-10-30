import pandas as pd

INPUT = "face_full_openvino_075_analysis.csv"
OUTPUT = "face_full_openvino_075_analysis.csv"

df = pd.read_csv(INPUT)
df["Filename"] = df["Filename"].str.replace("face_full_075","face_full",regex=False)
df.to_csv(OUTPUT, index=False, encoding="utf-8")
