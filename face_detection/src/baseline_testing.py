import numpy as np
import cv2
from insightface.app import FaceAnalysis
import os

app = FaceAnalysis(name="buffalo_l")  # pretrenirani InsightFace model
app.prepare(ctx_id=0)

def get_embedding(img):
    faces = app.get(img)
    if len(faces) == 0:
        return None
    return faces[0].embedding

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

def evaluate(database, test_path, unknown_path, threshold=0.6):
    correct = 0
    total = 0
    false_reject = 0
    false_accept = 0


    for person in os.listdir(test_path):
        folder = os.path.join(test_path, person)

        for img_name in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, img_name))
            emb = get_embedding(img)

            if emb is None:
                continue

            best_match = None
            best_score = -1

            for name, db_emb in database.items():
                score = cosine_similarity(emb, db_emb)

                if score > best_score:
                    best_score = score
                    best_match = name

            total += 1

            if best_match == person and best_score >= threshold:
                correct += 1
            else:
                false_reject += 1