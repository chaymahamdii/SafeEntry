import cv2
from datetime import datetime
from detect import detect_faces_in_frame
from liveness import check_liveness

photo_saved = False

print("Démarrage de la webcam...")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_faces_in_frame(frame)

    for (startX, startY, endX, endY) in faces:
        face_roi = frame[startY:endY, startX:endX]

        is_alive = check_liveness(face_roi)

        if is_alive:
            label = "Visage vivant"
            color = (0, 255, 0)

            if not photo_saved:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"static/real_face_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Photo sauvegardée : {filename}")
                photo_saved = True
        else:
            label = "Pas vivant / Ecran"
            color = (0, 0, 255)

        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Détection + Vivacité (SSD + MediaPipe)", frame)

    if cv2.waitKey(1) == 27:  # Échap pour quitter
        break

cap.release()
cv2.destroyAllWindows()
