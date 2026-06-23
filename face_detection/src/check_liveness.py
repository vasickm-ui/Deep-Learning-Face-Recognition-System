import onnxruntime as ort
import cv2
import numpy as np


liveness_session = ort.InferenceSession("models/anti-spoof/minifasnet_v2.onnx")
liveness_input = liveness_session.get_inputs()[0].name

def check_liveness(face, threshold=0.5):

    face = cv2.resize(face, (80, 80))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32) / 255.0
    face = np.transpose(face, (2, 0, 1))
    face = np.expand_dims(face, axis=0)

    out = liveness_session.run(None, {liveness_input: face})[0][0]

    # softmax
    exp = np.exp(out - np.max(out))
    probs = exp / exp.sum()

    spoof_score = probs[0] + probs[1]
    live_score = probs[2]


    score = live_score - spoof_score

    print("LIVE:", live_score)
    print("SPOOF:", spoof_score)
    print("SCORE:", score)

    is_live = score > threshold

    return is_live, float(score)