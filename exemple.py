import imageio.v3 as iio
import cv2
import numpy as np

# ✅ URL corrigée avec slashes
RTSP_URL = "rtsp://admin:L282777A@10.0.22.50:554/cam/realmonitor?channel=1&subtype=1&unicast=true&proto=Onvif"

# ✅ Lancement du flux
reader = iio.imiter(
    RTSP_URL,
    plugin="ffmpeg",
    input_params=["-rtsp_transport", "tcp", "-buffer_size", "1024000"]
)

print("[INFO] Stream RTSP via FFmpeg lancé...")

try:
    for frame in reader:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("RTSP Stream", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("[INFO] Arrêté par l'utilisateur.")
finally:
    reader.close()
    cv2.destroyAllWindows()
