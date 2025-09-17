import cv2
import os
import numpy as np
import face_recognition
from tensorflow.keras.models import load_model
import dlib
import time

# === Chargement modèle anti-spoofing ===
spoof_model = load_model("anti_spoof_model.h5", compile=False)

# === Chargement shape predictor dlib ===
predictor_path = "shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)

# === Paramètres ===
TOLERANCE = 0.3
KNOWN_FACES_DIR = "known_faces"
MIN_FACE_AREA = 30000

OBSERVATION_TIME = 5         # secondes à observer
MIN_FRAMES = 20              # nombre minimal de frames pendant l'observation

MOVEMENT_THRESHOLD = 3          # pixels déplacement minimal pour considérer mouvement global
BRIGHTNESS_CHANGE_THRESHOLD = 2  # variation luminosité minimale
MOTION_THRESHOLD = 10000          # seuil strict pour mouvement interne (différence entre textures)

MIN_REAL_FACE_SCORE = 0.7         # au moins 70% frames avec mouvement global ou variation luminosité
MIN_INTERNAL_MOTION_SCORE = 0.6   # au moins 60% frames avec mouvement interne détecté
SPOOF_MODEL_THRESHOLD = 0.3       # seuil strict modèle anti-spoofing (<0.3 = réel)

EAR_THRESHOLD = 0.25             # seuil EAR pour détecter œil fermé
CONSEC_FRAMES_BLINK = 2          # frames consécutives pour valider clignement

# === Chargement des visages connus ===
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

# Fonction pour calculer EAR
def eye_aspect_ratio(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# === Initialisation webcam ===
cap = cv2.VideoCapture(0)

prev_position = None
prev_brightness = None
prev_face_img = None

start_time = None
frames_analyzed = 0
movement_frames = 0
brightness_change_frames = 0
internal_motion_frames = 0
spoof_predictions = []

eye_closed_frames = 0
blink_detected = False
total_blinks = 0

verdict_given = False
verdict = None
verdict_reason = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    if len(face_locations) == 0:
        # Reset si pas de visage détecté
        start_time = None
        frames_analyzed = 0
        movement_frames = 0
        brightness_change_frames = 0
        internal_motion_frames = 0
        spoof_predictions = []
        verdict_given = False
        verdict = None
        verdict_reason = ""
        prev_position = None
        prev_brightness = None
        prev_face_img = None
        eye_closed_frames = 0
        blink_detected = False
        total_blinks = 0

        cv2.putText(frame, "Aucun visage détecté", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Liveness & Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # On prend seulement le premier visage détecté
    (top, right, bottom, left) = face_locations[0]
    encoding = face_encodings[0]

    # Reconnaissance faciale
    name = "Unknown"
    matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=TOLERANCE)
    face_distances = face_recognition.face_distance(known_encodings, encoding)
    if True in matches:
        best_match_index = np.argmin(face_distances)
        name = known_names[best_match_index]

    # Ajustement coordonnées (car frame réduit au début)
    top *= 2
    right *= 2
    bottom *= 2
    left *= 2

    face_width = right - left
    face_height = bottom - top
    face_area = face_width * face_height

    if face_area < MIN_FACE_AREA:
        cv2.putText(frame, "Visage trop petit", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Liveness & Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    face_img = frame[top:bottom, left:right]

    if face_img.size == 0:
        cv2.imshow("Liveness & Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Modèle anti-spoofing
    resized_face = cv2.resize(face_img, (224, 224))
    normalized_face = resized_face.astype("float32") / 255.0
    input_face = np.expand_dims(normalized_face, axis=0)
    prediction = spoof_model.predict(input_face)[0][0]
    spoof_predictions.append(prediction)

    # Calcul luminosité moyenne et position actuelle
    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    gray_face = cv2.resize(gray_face, (100, 100))  # taille fixe pour absdiff
    mean_brightness = np.mean(gray_face)
    current_position = (left, top, right, bottom)

    # Détection mouvements
    movement_detected = False
    brightness_changed = False
    internal_motion = False

    if prev_position is not None and prev_brightness is not None and prev_face_img is not None:
        dx = abs(current_position[0] - prev_position[0])
        dy = abs(current_position[1] - prev_position[1])
        movement_detected = (dx > MOVEMENT_THRESHOLD) or (dy > MOVEMENT_THRESHOLD)

        brightness_diff = abs(mean_brightness - prev_brightness)
        brightness_changed = brightness_diff > BRIGHTNESS_CHANGE_THRESHOLD

        diff = cv2.absdiff(gray_face, prev_face_img)
        motion_score = np.sum(diff)
        internal_motion = motion_score > MOTION_THRESHOLD
    else:
        motion_score = 0

    prev_position = current_position
    prev_brightness = mean_brightness
    prev_face_img = gray_face.copy()

    # === Detection clignement yeux avec dlib ===
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = face_detector(gray_frame, 0)

    if rects:
        rect = rects[0]
        shape = shape_predictor(gray_frame, rect)

        left_eye_pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(36, 42)])
        right_eye_pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(42, 48)])

        left_ear = eye_aspect_ratio(left_eye_pts)
        right_ear = eye_aspect_ratio(right_eye_pts)
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < EAR_THRESHOLD:
            eye_closed_frames += 1
        else:
            if eye_closed_frames >= CONSEC_FRAMES_BLINK:
                total_blinks += 1
            eye_closed_frames = 0

    if start_time is None:
        start_time = time.time()

    frames_analyzed += 1
    if movement_detected:
        movement_frames += 1
    if brightness_changed:
        brightness_change_frames += 1
    if internal_motion:
        internal_motion_frames += 1

    elapsed = time.time() - start_time

    # Verdict final strict après observation
    if elapsed >= OBSERVATION_TIME and not verdict_given and frames_analyzed >= MIN_FRAMES:
        avg_spoof_pred = np.mean(spoof_predictions)
        real_face_score = (movement_frames + brightness_change_frames) / frames_analyzed
        internal_motion_score = internal_motion_frames / frames_analyzed

        # Le verdict inclut la détection de clignement
        if (internal_motion_score >= MIN_INTERNAL_MOTION_SCORE and
            real_face_score >= MIN_REAL_FACE_SCORE and
            avg_spoof_pred < SPOOF_MODEL_THRESHOLD and
            total_blinks >= 1):
            verdict = "Visage réel détecté"
            verdict_reason = (f"Mouvements: {movement_frames}/{frames_analyzed}, "
                              f"Variations luminosité: {brightness_change_frames}/{frames_analyzed}, "
                              f"Mouvements internes: {internal_motion_frames}/{frames_analyzed}, "
                              f"Clignements yeux: {total_blinks}, "
                              f"Score modèle: {avg_spoof_pred:.2f}")
        else:
            verdict = "Visage spoof détecté"
            verdict_reason = (f"Mouvements: {movement_frames}/{frames_analyzed}, "
                              f"Variations luminosité: {brightness_change_frames}/{frames_analyzed}, "
                              f"Mouvements internes: {internal_motion_frames}/{frames_analyzed}, "
                              f"Clignements yeux: {total_blinks}, "
                              f"Score modèle: {avg_spoof_pred:.2f}")

        verdict_given = True

    # Affichage
    if verdict_given:
        color = (0, 255, 0) if "réel" in verdict.lower() else (0, 0, 255)
        display_text = f"{name} ({verdict})"
        cause_text = f"Cause: {verdict_reason}"
        cv2.putText(frame, display_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, cause_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    else:
        remaining = int(OBSERVATION_TIME - elapsed)
        cv2.putText(frame, f"Observation en cours... {remaining}s restantes",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Affichage debug mouvement interne et clignements
    cv2.putText(frame, f"Mouvement interne visage: {motion_score:.0f}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)
    cv2.putText(frame, f"Clignements yeux: {total_blinks}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)

    cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 2)
    cv2.imshow("Liveness & Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
