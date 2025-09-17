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
TOLERANCE = 0.4
KNOWN_FACES_DIR = "known_faces"
MIN_FACE_AREA = 30000

OBSERVATION_TIME = 5    # Durée observation (secondes)
MIN_FRAMES = 10         # Nombre minimal frames avant verdict

BRIGHTNESS_CHANGE_THRESHOLD = 10  
EYE_MOTION_THRESHOLD = 1500
NOSE_TURN_THRESHOLD = 7  # Seuil déplacement nez horizontal (en pixels)

MIN_INTERNAL_MOTION_SCORE = 0.6  # au moins 60% frames avec mouvement rotation tête
MIN_EYE_MOTION_SCORE = 0.6       # au moins 60% frames avec mouvement yeux
SPOOF_MODEL_THRESHOLD = 0.3      # seuil anti-spoofing (inférieur = réel)

# === Chargement visages connus ===
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

# === Initialisation webcam ===
cap = cv2.VideoCapture(0)

prev_brightness = None
prev_left_eye = None
prev_right_eye = None
prev_nose_x = None

start_time = None
frames_analyzed = 0
brightness_change_frames = 0
internal_motion_frames = 0
eye_motion_frames = 0
spoof_predictions = []

verdict_given = False
verdict = None
verdict_reason = ""

def get_eye_region(shape, left=True):
    points = [36, 37, 38, 39, 40, 41] if left else [42, 43, 44, 45, 46, 47]
    coords = np.array([[shape.part(p).x, shape.part(p).y] for p in points])
    x, y, w, h = cv2.boundingRect(coords)
    return x, y, w, h

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    if len(face_locations) == 0:
        # Reset si aucun visage détecté
        start_time = None
        frames_analyzed = 0
        brightness_change_frames = 0
        internal_motion_frames = 0
        eye_motion_frames = 0
        spoof_predictions = []
        verdict_given = False
        verdict = None
        verdict_reason = ""
        prev_brightness = None
        prev_left_eye = None
        prev_right_eye = None
        prev_nose_x = None

        cv2.putText(frame, "Aucun visage detecte", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Liveness & Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # On prend le premier visage détecté
    (top, right, bottom, left) = face_locations[0]
    encoding = face_encodings[0]

    # Reconnaissance faciale
    name = "Unknown"
    matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=TOLERANCE)
    face_distances = face_recognition.face_distance(known_encodings, encoding)
    if True in matches:
        best_match_index = np.argmin(face_distances)
        name = known_names[best_match_index]

    # Ajustement coordonnées pour cadre original
    top *= 2
    right *= 2
    bottom *= 2
    left *= 2

    face_width = right - left
    face_height = bottom - top
    face_area = face_width * face_height

    face_img = frame[top:bottom, left:right]
    if face_img.size == 0:
        cv2.imshow("Liveness & Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Prédiction anti-spoofing
    resized_face = cv2.resize(face_img, (224,224))
    normalized_face = resized_face.astype("float32") / 255.0
    input_face = np.expand_dims(normalized_face, axis=0)
    prediction = spoof_model.predict(input_face)[0][0]
    spoof_predictions.append(prediction)

    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    gray_face = cv2.resize(gray_face, (100,100))
    mean_brightness = np.mean(gray_face)

    brightness_changed = False
    internal_motion = False
    eye_motion = False

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = face_detector(gray_frame, 0)

    if prev_brightness is not None and rects:
        # Variation luminosité
        brightness_diff = abs(mean_brightness - prev_brightness)
        brightness_changed = brightness_diff > BRIGHTNESS_CHANGE_THRESHOLD

        # Landmarks
        rect = rects[0]
        shape = shape_predictor(gray_frame, rect)

        # Mouvement rotation tête (nez)
        current_nose_x = shape.part(30).x
        if prev_nose_x is not None:
            diff_nose_x = abs(current_nose_x - prev_nose_x)
            internal_motion = diff_nose_x > NOSE_TURN_THRESHOLD
        else:
            internal_motion = False

        prev_nose_x = current_nose_x

        # Mouvement yeux
        lx, ly, lw, lh = get_eye_region(shape, left=True)
        rx, ry, rw, rh = get_eye_region(shape, left=False)

        left_eye_img = gray_frame[ly:ly+lh, lx:lx+lw]
        right_eye_img = gray_frame[ry:ry+rh, rx:rx+rw]

        try:
            left_eye_img = cv2.resize(left_eye_img, (30,15))
            right_eye_img = cv2.resize(right_eye_img, (30,15))
        except:
            left_eye_img = None
            right_eye_img = None

        if prev_left_eye is not None and prev_right_eye is not None and left_eye_img is not None and right_eye_img is not None:
            diff_left = cv2.absdiff(left_eye_img, prev_left_eye)
            diff_right = cv2.absdiff(right_eye_img, prev_right_eye)
            eye_motion_score = np.sum(diff_left) + np.sum(diff_right)
            eye_motion = eye_motion_score > EYE_MOTION_THRESHOLD
        else:
            eye_motion_score = 0
    else:
        eye_motion_score = 0

    prev_brightness = mean_brightness
    prev_left_eye = left_eye_img if 'left_eye_img' in locals() else None
    prev_right_eye = right_eye_img if 'right_eye_img' in locals() else None

    if start_time is None:
        start_time = time.time()

    frames_analyzed += 1
    if brightness_changed:
        brightness_change_frames += 1
    if internal_motion:
        internal_motion_frames += 1
    if eye_motion:
        eye_motion_frames += 1

    elapsed = time.time() - start_time

    # Verdict final après observation stricte
    if elapsed >= OBSERVATION_TIME and not verdict_given and frames_analyzed >= MIN_FRAMES:
        avg_spoof_pred = np.mean(spoof_predictions)
        internal_motion_score = internal_motion_frames / frames_analyzed
        eye_motion_score_ratio = eye_motion_frames / frames_analyzed

        if (internal_motion_score >= MIN_INTERNAL_MOTION_SCORE and
            eye_motion_score_ratio >= MIN_EYE_MOTION_SCORE and
            avg_spoof_pred < SPOOF_MODEL_THRESHOLD):
            verdict = "Visage réel détecté"
            verdict_reason = (f"Variations luminosité: {brightness_change_frames}/{frames_analyzed}, "
                              f"Mouvements rotation tête: {internal_motion_frames}/{frames_analyzed}, "
                              f"Mouvements yeux: {eye_motion_frames}/{frames_analyzed}, "
                              f"Score modèle: {avg_spoof_pred:.2f}")
        else:
            verdict = "Visage spoof détecté"
            verdict_reason = (f"Variations luminosité: {brightness_change_frames}/{frames_analyzed}, "
                              f"Mouvements rotation tête: {internal_motion_frames}/{frames_analyzed}, "
                              f"Mouvements yeux: {eye_motion_frames}/{frames_analyzed}, "
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

    cv2.putText(frame, f"Mouvement rotation tête: {internal_motion_frames}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)
    cv2.putText(frame, f"Mouvement yeux: {eye_motion_frames}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)

    cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 2)
    cv2.imshow("Liveness & Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
