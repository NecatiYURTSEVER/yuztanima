import cv2
import os

# Yüz tanıma modeli yükleme
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Kamera başlatma
cap = cv2.VideoCapture(6)

# Klasör oluşturma
if not os.path.exists('faces'):
    os.makedirs('faces')

while True:
    # Kameradan bir kare al
    ret, frame = cap.read()

    # Görüntüyü griye çevir
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüzleri tespit etmek için haarcascade kullan
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Yüz bölgesini griye çevir ve boyutlandır
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (200, 200), interpolation=cv2.INTER_LINEAR)

        # Yüzü kaydet
        cv2.imwrite("faces/face_{}.jpg".format(len(os.listdir("faces"))+1), roi_gray)

        # Yüzü dikdörtgenle çerçeveleme
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Kamerayı gösterme
    cv2.imshow('Video', frame)

    # Çıkış için q tuşuna basma kontrolü
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı serbest bırakma
cap.release()
cv2.destroyAllWindows()
