import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Charger le modèle entraîné
model = tf.keras.models.load_model('anti_spoof_model.h5')

# Initialiser MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

def preprocess(face_img):
    img = cv2.resize(face_img, (224, 224))
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

cap = cv2.VideoCapture(0)

print("Appuie sur 'Échap' pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Détection des visages avec MediaPipe
    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape

            x1 = int(max(bboxC.xmin * w, 0))
            y1 = int(max(bboxC.ymin * h, 0))
            x2 = int(min((bboxC.xmin + bboxC.width) * w, w))
            y2 = int(min((bboxC.ymin + bboxC.height) * h, h))

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            input_data = preprocess(face)
            pred = model.predict(input_data)[0][0]

            # Tes classes : {'live': 0, 'spoof': 1}
            # Donc live = 0, spoof = 1, on teste si pred < 0.5 → live
            if pred < 0.5:
                label = "Vrai visage"
                color = (0, 255, 0)
            else:
                label = "Spoof (faux visage)"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Test Anti-Spoofing en Live", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Touche Échap pour quitter
        break

cap.release()
cv2.destroyAllWindows()
