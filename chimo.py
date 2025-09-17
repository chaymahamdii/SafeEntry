import cv2
import os
import numpy as np
import face_recognition
from tensorflow.keras.models import load_model
import time
import dlib

# CONFIGURATION
ESP32_CAM_STREAM_URL = "http://10.0.22.30:81/stream"
TOLERANCE = 0.4
KNOWN_FACES_DIR = "known_faces"
OBSERVATION_TIME = 10
MIN_FRAMES = 20
SPOOF_MODEL_THRESHOLD = 0.3
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 1

# CHARGEMENT DES MODÈLES
spoof_model = load_model("anti_spoof_model.h5", compile=False)
predictor_path = "shape_predictor_68_face_landmarks.dat"
shape_predictor = dlib.shape_predictor(predictor_path)
face_detector = dlib.get_frontal_face_detector()

# CHARGER LES VISAGES CONNUS
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

# VARIABLES
eye_closed_frames = 0
total_blinks = 0
spoof_predictions = []
start_time = None
frames_analyzed = 0
verdict_given = False
verdict = None
verdict_reason = ""

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

cap = cv2.VideoCapture(ESP32_CAM_STREAM_URL)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    if len(face_locations) == 0:
        # Pas de visage détecté => reset
        start_time = None
        spoof_predictions = []
        eye_closed_frames = 0
        total_blinks = 0
        frames_analyzed = 0
        verdict_given = False
        verdict = None
        verdict_reason = ""
        cv2.imshow("Video Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # On prend le premier visage détecté
    (top, right, bottom, left) = face_locations[0]
    encoding = face_encodings[0]

    # Reconnaissance du nom
    name = "Unknown"
    matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=TOLERANCE)
    face_distances = face_recognition.face_distance(known_encodings, encoding)
    if True in matches:
        best_match_index = np.argmin(face_distances)
        name = known_names[best_match_index]

    # Ajuster échelle car face_recognition travaille sur image réduite
    top *= 2
    right *= 2
    bottom *= 2
    left *= 2

    face_img = frame[top:bottom, left:right]
    if face_img.size == 0:
        continue

    # Anti-spoofing
    resized_face = cv2.resize(face_img, (224, 224))
    normalized_face = resized_face.astype("float32") / 255.0
    input_face = np.expand_dims(normalized_face, axis=0)
    prediction = spoof_model.predict(input_face)[0][0]
    spoof_predictions.append(prediction)

    # Détection clignement d’yeux avec dlib
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dlib_faces = face_detector(gray)
    if dlib_faces:
        shape = shape_predictor(gray, dlib_faces[0])
        left_eye = np.array([(shape.part(i).x, shape.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(shape.part(i).x, shape.part(i).y) for i in range(42, 48)])

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        if ear < EAR_THRESHOLD:
            eye_closed_frames += 1
        else:
            if eye_closed_frames >= CONSEC_FRAMES:
                total_blinks += 1
            eye_closed_frames = 0

    if start_time is None:
        start_time = time.time()

    frames_analyzed += 1
    elapsed = time.time() - start_time

    # Affichage rectangle visage + nom
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(frame, name, (left, top - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Verdict après observation
    if elapsed >= OBSERVATION_TIME and not verdict_given and frames_analyzed >= MIN_FRAMES:
        avg_spoof_pred = np.mean(spoof_predictions)

        if avg_spoof_pred < SPOOF_MODEL_THRESHOLD and total_blinks >= 1:
            verdict = "Visage reel detecte"
            color = (0, 255, 0)
        else:
            verdict = "Visage spoof detecte"
            color = (0, 0, 255)

        verdict_reason = f"Score modèle: {avg_spoof_pred:.2f}, Clignements: {total_blinks}"
        verdict_given = True

    # Affichage verdict et détails
    if verdict_given and verdict is not None:
        cv2.putText(frame, verdict, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y0 = bottom + 25
        for i, line in enumerate(verdict_reason.split(", ")):
            y = y0 + i * 25
            cv2.putText(frame, line, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

    cv2.imshow("Video Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
