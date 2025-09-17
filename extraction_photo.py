import cv2
import mediapipe as mp
import os

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3)

def extract_faces(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, filename)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Impossible de lire l'image {filename}")
                continue

            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = image.shape
                    x1 = int(bboxC.xmin * w)
                    y1 = int(bboxC.ymin * h)
                    x2 = x1 + int(bboxC.width * w)
                    y2 = y1 + int(bboxC.height * h)

                    # Clamp des coordonnées pour qu'elles soient dans l'image
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)

                    # Vérifier que les coordonnées forment une zone valide
                    if x2 <= x1 or y2 <= y1:
                        print(f"Zone invalide pour {filename} : ({x1},{y1})-({x2},{y2})")
                        continue

                    face = image[y1:y2, x1:x2]

                    if face.size == 0:
                        print(f"Visage vide extrait dans {filename}")
                        continue

                    face_filename = os.path.join(output_dir, f'face_{count}.jpg')
                    cv2.imwrite(face_filename, face)
                    count += 1
                    print(f'Visage extrait et sauvegardé : {face_filename}')
            else:
                print(f"Aucun visage détecté dans {filename}")

extract_faces('data/live/photos', 'data/live/faces_from_photos')
extract_faces('data/spoof/photos', 'data/spoof/faces_from_photo')
