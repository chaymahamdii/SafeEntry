
import cv2
import mediapipe as mp
import os

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def extract_faces_from_video(video_path, output_dir, max_frames=100):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x1 = int(bboxC.xmin * w)
                y1 = int(bboxC.ymin * h)
                x2 = x1 + int(bboxC.width * w)
                y2 = y1 + int(bboxC.height * h)
                face = frame[y1:y2, x1:x2]
                face_filename = os.path.join(output_dir, f'face_{count}.jpg')
                cv2.imwrite(face_filename, face)
                count += 1
                print(f'Visage extrait de la vidéo et sauvegardé : {face_filename}')
        if count >= max_frames:
            break
    cap.release()

# Extraire faces vidéos live
live_videos_dir = 'data/live/videos'
live_faces_dir = 'data/live/faces_from_videos'
os.makedirs(live_faces_dir, exist_ok=True)
for video_file in os.listdir(live_videos_dir):
    if video_file.lower().endswith('.avi'):
        extract_faces_from_video(os.path.join(live_videos_dir, video_file), live_faces_dir)

# Extraire faces vidéos spoof
spoof_videos_dir = 'data/spoof/videos'
spoof_faces_dir = 'data/spoof/faces_from_videos'
os.makedirs(spoof_faces_dir, exist_ok=True)
for video_file in os.listdir(spoof_videos_dir):
    if video_file.lower().endswith('.avi'):
        extract_faces_from_video(os.path.join(spoof_videos_dir, video_file), spoof_faces_dir)
