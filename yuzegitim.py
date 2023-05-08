import cv2
import os
import numpy as np

# Yüz tanıma için CascadeClassifier kullanacağız
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

# Veri setinin bulunduğu dizin
data_dir = 'faces'

# Veri setindeki yüzleri tutacak listeler
faces = []
labels = []

# Veri setindeki tüm kişilerin klasörlerini alıyoruz
people = [person for person in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, person))]

# Her kişinin yüzlerini okuyoruz ve listelere ekliyoruz
for i, person in enumerate(people):
    person_dir = os.path.join(data_dir, person)
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in face:
            roi_gray = gray[y:y + h, x:x + w]
            faces.append(roi_gray)
            labels.append(i)

# Eğitim için LBPH tanıyıcıyı kullanıyoruz
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Yüzleri ve etiketleri eğitiyoruz
recognizer.train(faces, np.array(labels))

# Eğitilmiş tanıyıcıyı kaydediyoruz
recognizer.save('trainer.yml')

print("Yüz tanıma eğitimi tamamlandı!")
