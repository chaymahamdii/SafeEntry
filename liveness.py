import cv2
import mediapipe as mp
from math import hypot
import numpy as np

# Initialisation de MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True)

# Indices des points des yeux
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def eye_aspect_ratio(landmarks, eye_indices):
    p1 = landmarks[eye_indices[1]]
    p2 = landmarks[eye_indices[5]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[4]]
    p5 = landmarks[eye_indices[0]]
    p6 = landmarks[eye_indices[3]]

    A = hypot(p2.x - p1.x, p2.y - p1.y)
    B = hypot(p4.x - p3.x, p4.y - p3.y)
    C = hypot(p6.x - p5.x, p6.y - p5.y)

    ear = (A + B) / (2.0 * C)
    return ear

def is_texture_real(gray):
    var = np.var(gray)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return var > 100 and laplacian > 50

def is_lighting_natural(gray):
    h = gray.shape[0]
    top = gray[:h//2, :]
    bottom = gray[h//2:, :]
    mean_top = np.mean(top)
    mean_bottom = np.mean(bottom)
    diff = abs(mean_top - mean_bottom)
    return diff > 5  # si la lumière est trop uniforme → écran probable

def has_no_screen_edges(gray):
    edges = cv2.Canny(gray, 100, 200)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    return lines is None or len(lines) < 10  # trop de lignes droites = bord d'écran

def is_3d_face(landmarks):
    nose_tip = landmarks[1].z
    left_cheek = landmarks[234].z
    right_cheek = landmarks[454].z
    # Sur un vrai visage, le nez est plus proche que les joues
    return nose_tip < left_cheek and nose_tip < right_cheek

def check_liveness(face_roi):
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    # Test 1 : texture
    if not is_texture_real(gray):
        print("❌ Texture suspecte → probablement un écran")
        return False

    # Test 2 : lumière
    if not is_lighting_natural(gray):
        print("❌ Lumière trop uniforme → probablement un écran")
        return False

    # Test 3 : bords
    if not has_no_screen_edges(gray):
        print("❌ Bords droits détectés → probablement un écran")
        return False

    # Test 4 : vérification 3D
    rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            if not is_3d_face(landmarks):
                print("❌ Visage plat détecté → probablement une image ou vidéo")
                return False

            # (facultatif) détection d'yeux fermés
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0
            if avg_ear < 0.20:
                print("ℹ️ Yeux fermés détectés")
            else:
                print("ℹ️ Yeux ouverts détectés")

        # Tous les tests sont passés ✅
        print("✅ Visage réel détecté")
        return True

    print("❌ Aucun visage détecté par MediaPipe")
    return False
