import cv2 
import os
import dlib 
import face recognition 
import time
from tensorflow.keras.models import load_model
import numpy  as np 

#chargements des modèles 

spoof_model = load_model("anti_spoof_model.h5", compile=False)
predictor_path = "shape_predictor_68_face_landmarks.dat"
shape_predictor = dlib.shape_predictor(predictor_path)
face_detector = dlib.get_frontal_face_detector()


