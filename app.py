from flask import Flask, render_template, request
import pandas as pd
import os

IMG_FOLDER = "static/face_full"
CSV = "face_full_openvino_analysis.csv"

app = Flask(__name__)
@app.route("/")
def dashboard():
    df = pd.read_csv(CSV)
    
    df["Gender"] = df["Gender"].str.capitalize()
    #df["ImagePath"] = df["Filename"].apply(lambda x: os.path.join(IMG_FOLDER,x))
    df["ImagePath"] = df["Filename"].apply(
        lambda x: os.path.join(IMG_FOLDER, x.replace("face_full/",""))
    )
    selected_gender = request.args.get("gender", "All")
    if selected_gender != "All":
        df = df[df["Gender"] == selected_gender]
    
    total_faces = len(pd.read_csv(CSV))
    gender_counts = pd.read_csv(CSV)["Gender"].value_counts().to_dict()
    gender_percent = {
        g: round((c/total_faces)*100, 2)for g, c in gender_counts.items()
    }
    
    data = df.to_dict(orient="records")
    return render_template("dashboard.html", faces=data,
                           selected_gender=selected_gender, gender_percent=gender_percent
                           , total_faces = total_faces)

if __name__ == "__main__":
    app.run(host = '0.0.0.0', port=1909, debug=False)