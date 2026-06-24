import onnxruntime as ort
import cv2
import numpy as np
from paths import PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models" / "anti-spoof" / "AntiSpoofing_print-replay_15_128.onnx"

liveness_session = ort.InferenceSession(
    MODEL_PATH
)

liveness_input = liveness_session.get_inputs()[0].name


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def check_liveness_binary(face, threshold=0.5):
    face = cv2.resize(face, (128, 128))   # IMPORTANT: model is 128x128
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    face = face.astype(np.float32) / 255.0  # safer for this model

    face = np.transpose(face, (2, 0, 1))  # CHW
    face = np.expand_dims(face, axis=0)

    out = liveness_session.run(None, {liveness_input: face})[0][0]


    probs = softmax(out)

    print(f"SPOOF prob: {probs[0]}")
    print(f"LIVE  prob: {probs[1]}")

    is_live = probs[1] > threshold

    return is_live

def check_liveness_print_replay(face, threshold):

    # preprocess (same as tested model: 128x128 CHW RGB)
    face = cv2.resize(face, (128, 128))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    face = face.astype(np.float32) / 255.0
    face = np.transpose(face, (2, 0, 1))
    face = np.expand_dims(face, axis=0)

    # inference
    out = liveness_session.run(None, {liveness_input: face})[0][0]

    # probabilities
    probs = softmax(out)

    real_prob = probs[0]
    print_prob = probs[1]
    replay_prob = probs[2]

    print(f"REAL  prob : {real_prob}")
    print(f"PRINT prob : {print_prob}")
    print(f"REPLAY prob: {replay_prob}")

    is_spoof = print_prob > threshold or replay_prob > threshold
    return is_spoof