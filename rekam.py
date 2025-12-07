import cv2, time, os

camera = 0
video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)

facedeteksi = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if facedeteksi.empty():
    print("❌ Gagal memuat file haarcascade!")
    exit()

id = input('Masukan id : ')
a = 0
os.makedirs('Database', exist_ok=True)

while True:
    a += 1
    check, frame = video.read()
    if not check:
        print("❌ Gagal membaca kamera.")
        break
    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = facedeteksi.detectMultiScale(abu, 1.3, 5)
    for (x, y, w, h) in wajah:
        cv2.imwrite(f'Database/User.{id}.{a}.jpg', abu[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if a > 29:
        print("✅ Pengambilan data wajah selesai.")
        break

video.release()
cv2.destroyAllWindows()