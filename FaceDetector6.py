import cv2

# Carregar imagem de referência
imgElon = cv2.imread('Eu.jpg')
imgElon_gray = cv2.cvtColor(imgElon, cv2.COLOR_BGR2GRAY)

# Inicializar webcam
cap = cv2.VideoCapture(0)

# Detector de rosto Haar
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ORB para pontos-chave
orb = cv2.ORB_create()

# Descritor da imagem de referência
kp1, des1 = None, None
faces_ref = face_cascade.detectMultiScale(imgElon_gray, 1.3, 5)
for (x, y, w, h) in faces_ref:
    roi_ref = imgElon_gray[y:y+h, x:x+w]
    kp1, des1 = orb.detectAndCompute(roi_ref, None)
    break  # usar apenas o primeiro rosto

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_face = gray[y:y+h, x:x+w]
        kp2, des2 = orb.detectAndCompute(roi_face, None)

        if des2 is not None and des1 is not None:
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            score = len(matches) / max(len(kp1), 1)  # porcentagem de correspondência

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Match: {score*100:.1f}%", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Comparacao", frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
