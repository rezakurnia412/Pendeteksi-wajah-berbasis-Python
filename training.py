import cv2, os
import numpy as np
from PIL import Image

detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

if detector.empty():
    print("❌ Gagal memuat file haarcascade!")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()

def getImagesWithLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(imageNp)
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y + h, x:x + w])
            Ids.append(Id)
    return faceSamples, Ids

faces, Ids = getImagesWithLabels('Database')
recognizer.train(faces, np.array(Ids))
os.makedirs('Database', exist_ok=True)
recognizer.save('Database/training.xml')

print("✅ Training selesai, file model disimpan di Database/training.xml")