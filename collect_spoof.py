import cv2
import os

# Créer les dossiers pour spoof si ce n'est pas encore fait
os.makedirs('data/spoof/photos', exist_ok=True)
os.makedirs('data/spoof/videos', exist_ok=True)

# Initialiser le compteur photos et vidéos avec les fichiers existants
existing_spoof_photos = [f for f in os.listdir('data/spoof/photos') if f.endswith('.jpg')]
photo_count = len(existing_spoof_photos)

existing_spoof_videos = [f for f in os.listdir('data/spoof/videos') if f.endswith('.avi')]
video_count = len(existing_spoof_videos)

recording = False

print("Appuie sur 'p' pour prendre une photo, 'v' pour commencer/arrêter la vidéo, 'q' pour quitter.")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Webcam Capture Spoof', frame)
    key = cv2.waitKey(1) & 0xFF

    # Prendre une photo spoof
    if key == ord('p'):
        photo_count += 1
        filename = f'data/spoof/photos/spoof_photo_{photo_count}.jpg'
        cv2.imwrite(filename, frame)
        print(f'Photo spoof enregistrée : {filename}')

    # Commencer/arrêter une vidéo spoof
    if key == ord('v'):
        if not recording:
            recording = True
            video_count += 1
            video_filename = f'data/spoof/videos/spoof_video_{video_count}.avi'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            print('Enregistrement vidéo spoof commencé...')
        else:
            recording = False
            out.release()
            print(f'Vidéo spoof enregistrée : {video_filename}')

    if recording:
        out.write(frame)

    # Quitter avec 'q'
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
