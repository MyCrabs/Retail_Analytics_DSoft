from openvino import Core
import cv2
import numpy as np

core = Core()
model_path = "weight/age-gender-recognition-retail-0013.xml"
compiled_model = core.compile_model(model_path, 'CPU')

age_output = compiled_model.output(0)
gender_output = compiled_model.output(1)

def predict_image(img):
    img = cv2.imread(img)
    img = cv2.resize(img, (62,62))
    img = img.transpose((2, 0, 1))[None, :]
    img = img.astype(np.float32)
    
    res = compiled_model([img])
    age = float(res[age_output][0][0][0][0] * 100)
    prob = float(res[gender_output][0][0][0][0])
    gender = "Female" if prob > 0.5 else "Male"

    return round(age, 1), gender


predict_image("input/thduy.jpg")