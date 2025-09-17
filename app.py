from flask import Flask, request, jsonify
import cv2
import numpy as np
from detect import detect_faces
from liveness import check_liveness

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    img = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img, cv2.IMREAD_COLOR)

    if not detect_faces(frame):
        return jsonify({"result": "Pas de visage détecté"})

    if not check_liveness(frame):
        return jsonify({"result": "Visage non vivant — Refus"})

    # Sinon, visage vivant détecté
    # ⚠️ ici tu peux ajouter la commande pour activer le relais
    return jsonify({"result": "Visage réel détecté — Accès autorisé"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

