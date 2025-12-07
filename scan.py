import cv2, time
import os
from PIL import Image

camera = 0
video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
facedeteksi = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('Database/training.xml')

if facedeteksi.empty():
    print("❌ Gagal memuat file haarcascade!")
    exit()

while True:
    check, frame = video.read()
    if not check:
        print("❌ Tidak dapat membaca kamera.")
        break

    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = facedeteksi.detectMultiScale(abu, 1.3, 5)
    for (x, y, w, h) in wajah:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, conf = recognizer.predict(abu[y:y + h, x:x + w])
        if   id == 1:
            nama = 'Reza akbar'
        elif id == 2:
            nama = 'ALDY PONCO'
        elif id == 3:
            nama = 'FATIH FADIL'
        elif id == 4:
            nama = 'Elon Musk'
        elif id == 5:
            nama = 'Ahmad Surya'
        elif id == 6:
            nama = 'Reza auliya'
        elif id == 7:
            nama = 'Daril Azkia'
        elif id == 8:
            nama = 'Haekal Nuaim'
        elif id == 9:
            nama = 'Ragadika Aziz Saputra'
        elif id == 10:
            nama = 'Sofyan Ahmadi'
        else:
            nama = 'Tidak dikenal'
        cv2.putText(frame, f"{nama} ({round(conf, 2)})", (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(1)
    if key == ord('a'):
        break

video.release()
cv2.destroyAllWindows()