import cv2
import os
import numpy as np
import face_recognition
from tensorflow.keras.models import load_model
import time
import dlib
import random
#models
spoof_model = load_model("anti_spoof_model.h5", compile=False)
predictor_path = "shape_predictor_68_face_landmarks.dat"
shape_predictor = dlib.shape_predictor(predictor_path)
face_detector = dlib.get_frontal_face_detector()
#parametres
TOLERANCE = 0.4
KNOWN_FACES_DIR = "known_faces"
MIN_FACE_AREA = 30000

OBSERVATION_TIME = 10
MIN_FRAMES = 20
SPOOF_MODEL_THRESHOLD = 0.3


ACTIVITIES = [
    "Tourne la tete a gauche",
    "Tourne la tete a droite",
    "Souris",
    #"Ouvre grand les yeux"
]
current_activity = random.choice(ACTIVITIES)
activity_done = False
#chargement des visages connus
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

#activer le webcam
cap = cv2.VideoCapture(0)

start_time = None
frames_analyzed = 0
spoof_predictions = []
verdict_given = False
verdict = None
verdict_reason = ""
#les activites
def detect_activity(activity, shape):
    if activity == "Tourne la tete a gauche":
        nose = shape.part(30).x
        chin = shape.part(8).x
        return nose - chin < -10
    elif activity == "Tourne la tete a droite":
        nose = shape.part(30).x
        chin = shape.part(8).x
        return nose - chin > 10
    elif activity == "Souris":
        mouth_width = shape.part(54).x - shape.part(48).x
        return mouth_width > 60
    #elif activity == "Ouvre grand les yeux":
        #left_eye_height = shape.part(41).y - shape.part(37).y
        #right_eye_height = shape.part(46).y - shape.part(43).y
        #return left_eye_height < -5 and right_eye_height < -5
    return False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_small_frame = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=0.5, fy=0.5), cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    if len(face_locations) == 0:
        start_time = None
        frames_analyzed = 0
        spoof_predictions = []
        verdict_given = False
        verdict = None
        activity_done = False
        current_activity = random.choice(ACTIVITIES)
        cv2.putText(frame, "Aucun visage detecte", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Liveness & Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    (top, right, bottom, left) = face_locations[0]
    encoding = face_encodings[0]

    name = "Unknown"
    matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.4)
    face_distances = face_recognition.face_distance(known_encodings, encoding)
    if True in matches:
        best_match_index = np.argmin(face_distances)
        name = known_names[best_match_index]

    top *= 2
    right *= 2
    bottom *= 2
    left *= 2

    face_img = frame[top:bottom, left:right]
    if face_img.size == 0:
        continue

    resized_face = cv2.resize(face_img, (224, 224))
    normalized_face = resized_face.astype("float32") / 255.0
    input_face = np.expand_dims(normalized_face, axis=0)
    prediction = spoof_model.predict(input_face)[0][0]
    spoof_predictions.append(prediction)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dlib_faces = face_detector(gray)
    if dlib_faces:
        shape = shape_predictor(gray, dlib_faces[0])
        if detect_activity(current_activity, shape):
            activity_done = True

    if start_time is None:
        start_time = time.time()

    frames_analyzed += 1
    elapsed = time.time() - start_time

    if elapsed >= OBSERVATION_TIME and not verdict_given and frames_analyzed >= MIN_FRAMES:
        avg_spoof_pred = np.mean(spoof_predictions)
        if activity_done and avg_spoof_pred < SPOOF_MODEL_THRESHOLD:
            verdict = "Visage reel detecte"
        else:
            verdict = "Visage spoof detecte"

        verdict_reason = f"Activite terminee: {activity_done}, Score modele: {avg_spoof_pred:.2f}"
        verdict_given = True

    if verdict_given:
        color = (0, 255, 0) if "reel" in verdict.lower() else (0, 0, 255)
        cv2.putText(frame, f"{name} ({verdict})", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Cause: {verdict_reason}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    else:
        remaining = int(OBSERVATION_TIME - elapsed)
        cv2.putText(frame, f"Fais ceci: {current_activity}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"{remaining}s restantes", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 2)
    cv2.imshow("Liveness & Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
