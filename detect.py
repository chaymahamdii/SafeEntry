import cv2
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

def detect_faces_in_frame(frame):
    h, w = frame.shape[:2]
    
   
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

   
    net.setInput(blob)
    detections = net.forward()

    faces = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX, endY))

    return faces
