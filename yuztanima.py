import cv2
import os
import numpy as np


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

names = os.listdir("faces")

cap = cv2.VideoCapture(6)


fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS:", fps)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:

        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (200, 200), interpolation=cv2.INTER_LINEAR)

        id_, result = recognizer.predict(roi_gray)

        if result < 50:
            name = names[id_]
            confidence = int(100*(1-(result)/300))
            cv2.putText(frame, name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(frame, str(confidence)+"%", (x+5, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 1)
        else:
            cv2.putText(frame, "Unknown", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    fps_text = "FPS: " + str(int(fps))
    cv2.putText(frame, fps_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
