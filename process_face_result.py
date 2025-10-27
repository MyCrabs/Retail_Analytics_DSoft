import pandas as pd
from collections import Counter
import re

CSV_IN = "face_result.csv"
CSV_OUT = "face_result_grouped.csv"

def extract_id(filename):
    match = re.match(r"id(\d+)_", filename)
    return int(match.group(1)) if match else None

def mode_gender(genders):
    if not genders:
        return None
    return Counter(genders).most_common(1)[0][0]

def aggregate_result(csv_path):
    df = pd.read_csv(csv_path)
    df["person_id"] = df["filename"].apply(extract_id)
    df = df.dropna(subset=["person_id"])
    grouped = []
    for pid, group in df.groupby("person_id"):
        ages = group["age"].dropna().astype(float).tolist()
        genders = group["dominant_gender"].dropna().tolist()
        avg_age = round(sum(ages) / len(ages), 1) if ages else None
        final_gender = mode_gender(genders)
        grouped.append({
            "person_id":int(pid),
            "avg_age": avg_age,
            "final_gender": final_gender,
            "num_samples": len(group)
        })
    out_df = pd.DataFrame(grouped)
    out_df.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    print(f"Kết quả tổng hợp đã lưu tại: {CSV_OUT}")

if __name__ =="__main__":
    aggregate_result(CSV_IN)