import cv2
import os

os.makedirs('data/live/photos', exist_ok=True)
os.makedirs('data/live/videos', exist_ok=True)

# Compter le nombre de photos existantes pour ne pas écraser
existing_photos = [f for f in os.listdir('data/live/photos') if f.endswith('.jpg')]
photo_count = len(existing_photos)

video_count = 0
recording = False

print("Appuie sur 'p' pour prendre une photo, 'v' pour commencer/arrêter une vidéo, 'q' pour quitter.")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Capture Live', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('p'):
        photo_count += 1
        filename = f'data/live/photos/photo_{photo_count}.jpg'
        cv2.imwrite(filename, frame)
        print(f'Photo live enregistrée : {filename}')

    if key == ord('v'):
        if not recording:
            recording = True
            video_count += 1
            video_filename = f'data/live/videos/video_{video_count}.avi'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            print('Enregistrement vidéo live commencé...')
        else:
            recording = False
            out.release()
            print(f'Vidéo live enregistrée : {video_filename}')

    if recording:
        out.write(frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
