import onnxruntime as ort
import cv2
import numpy as np


liveness_session = ort.InferenceSession("models/anti-spoof/minifasnet_v2.onnx")
liveness_input = liveness_session.get_inputs()[0].name

def check_liveness(face, threshold):
    face = cv2.resize(face, (80, 80))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32) / 255.0
    face = np.transpose(face, (2, 0, 1))
    face = np.expand_dims(face, axis=0)

    out = liveness_session.run(None, {liveness_input: face})[0]

    # softmax
    e = np.exp(out[0] - np.max(out[0]))
    probs = e / e.sum()

    live_score = probs[0]
    spoof_score = probs[1] + probs[2]

    is_live = live_score > threshold

    return is_live, float(live_score)