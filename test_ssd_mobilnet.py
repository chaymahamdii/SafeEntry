import os
import cv2
import numpy as np
import face_recognition
import tensorflow as tf

# --- Chargement du modèle SSD Mobilenet ---
prototxt_path = "deploy.prototxt"
weights_path = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)

# --- Chargement du modèle anti-spoofing ---
anti_spoof_model = tf.keras.models.load_model('anti_spoof_model.h5')

# --- Chargement des visages connus depuis known_faces ---
known_faces_dir = "known_faces"
known_face_encodings = []
known_face_names = []

print("[INFO] Chargement des visages connus...")
for person_name in os.listdir(known_faces_dir):
    person_dir = os.path.join(known_faces_dir, person_name)
    if not os.path.isdir(person_dir):
        continue
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            known_face_encodings.append(encodings[0])
            known_face_names.append(person_name)
        else:
            print(f"[WARNING] Pas de visage détecté dans {image_path}")

print(f"[INFO] {len(known_face_encodings)} visages chargés.")

# --- Paramètres ---
CONFIDENCE_THRESHOLD = 0.5
RECOGNITION_TOLERANCE = 0.5

# --- Ouvrir la webcam ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_THRESHOLD:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Extraire visage détecté
            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            # Anti-spoofing
            face_resized = cv2.resize(face, (224, 224))
            face_norm = face_resized.astype("float32") / 255.0
            face_input = np.expand_dims(face_norm, axis=0)
            pred = anti_spoof_model.predict(face_input)[0][0]
            label = "Real" if pred > 0.5 else "Spoof"

            if label == "Spoof":
                name = "Spoof"
                color = (0, 0, 255)
            else:
                # Reconnaissance faciale
                rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb_face)
                name = "Unknown"
                color = (255, 0, 0)

                if len(encodings) > 0:
                    face_encoding = encodings[0]
                    distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    min_distance = min(distances)
                    if min_distance < RECOGNITION_TOLERANCE:
                        index = np.argmin(distances)
                        name = known_face_names[index]
                        color = (0, 255, 0)

            # Affichage
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, f"{name} ({label})", (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Face Detection & Anti-Spoofing", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
