import cv2
import os
import numpy as np
import face_recognition
from tensorflow.keras.models import load_model

# Charger le modèle anti-spoofing
spoof_model = load_model("anti_spoof_model.h5")

# Paramètres
TOLERANCE = 0.4
KNOWN_FACES_DIR = "known_faces"

# Charger les visages connus
known_encodings = []
known_names = []

print("[INFO] Chargement des visages connus...")
for person_name in os.listdir(KNOWN_FACES_DIR):
    person_path = os.path.join(KNOWN_FACES_DIR, person_name)
    for filename in os.listdir(person_path):
        image_path = os.path.join(person_path, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(person_name)

print(f"[INFO] {len(known_encodings)} visages connus chargés.")

# Ouvrir la webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        name = "Unknown"

        # Comparaison avec les visages connus
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.4)
        face_distances = face_recognition.face_distance(known_encodings, encoding)

        if True in matches:
            best_match_index = np.argmin(face_distances)
            name = known_names[best_match_index]

        # Spoof Detection
        face_img = frame[top*2:bottom*2, left*2:right*2]  # *2 car l'image est réduite
        if face_img.size == 0:
            continue

        resized_face = cv2.resize(face_img, (224, 224))
        normalized_face = resized_face.astype("float32") / 255.0
        input_face = np.expand_dims(normalized_face, axis=0)

        prediction = spoof_model.predict(input_face)[0][0]
        label = "Real" if prediction < 0.5 else "Spoof"

        # Texte à afficher
        display_text = f"{name} ({label})"

        color = (0, 255, 0) if label == "Real" and name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left*2, top*2), (right*2, bottom*2), color, 2)
        cv2.putText(frame, display_text, (left*2, top*2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Liveness & Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
