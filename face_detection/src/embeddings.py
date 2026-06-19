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

#this method calculates avg emedings per person and stores it in dict
def calculate_avg_embeddings(enroll_path, n):
    emb_data = {}
    #going through folders
    for person in os.listdir(enroll_path):
        person_name_folder = os.path.join(enroll_path, person)
        embeddings = []

        #going through image files
        for img_name in os.listdir(person_name_folder):

            img_path = os.path.join(person_name_folder, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Cant load image with path {img_path}")
                exit()

            emb = get_embedding(img)

            if emb is None:
                continue

            embeddings.append(emb)
            if len(embeddings) >= n:
                break

        
        avg_emb = np.mean(embeddings, axis=0)
        emb_data[person] = avg_emb
    
    return emb_data

def quality_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = min(blur / 150.0, 1.0)  # normalize

    brightness = np.mean(gray)
    if brightness < 40:
        brightness_score = brightness / 40.0
    elif brightness > 220:
        brightness_score = (255 - brightness) / 35.0
    else:
        brightness_score = 1.0

    brightness_score = np.clip(brightness_score, 0.0, 1.0)

    contrast = gray.std()
    contrast_score = min(contrast / 40.0, 1.0)

    quality = (
        0.5 * blur_score +
        0.25 * brightness_score +
        0.25 * contrast_score
    )

    return quality

def weighted_avg_embeddings(enroll_path, n):
    emb_data = {}

    for person in os.listdir(enroll_path):
        person_name_folder = os.path.join(enroll_path, person)

        embeddings = []
        weights = []

        for img_name in os.listdir(person_name_folder):
            img_path = os.path.join(person_name_folder, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Cant load image with path {img_path}")
                continue

            emb = get_embedding(img)
            if emb is None:
                continue

            q = quality_score(img)

            # hard reject if too bad
            if q < 0.25:
                continue

            embeddings.append(emb)
            weights.append(q)

            if len(embeddings) >= n:
                break

        if len(embeddings) == 0:
            print(f"No valid embeddings for {person}")
            continue

        embeddings = np.array(embeddings)
        weights = np.array(weights)

        # normalize weights
        weights = weights / np.sum(weights)

        avg_emb = np.sum(embeddings * weights[:, None], axis=0)

        emb_data[person] = avg_emb

    return emb_data