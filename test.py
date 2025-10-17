from deepface import DeepFace
import cv2

img = cv2.imread("input/test_age.png")
res = DeepFace.analyze(img, actions=["gender", "age"], enforce_detection=False)

print(res[0]["dominant_gender"], res[0]["age"])