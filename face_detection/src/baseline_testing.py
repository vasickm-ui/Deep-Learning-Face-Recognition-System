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
def calculate_avg_embeddings(enroll_path):
    emb_data = {}
    for person in os.listdir(enroll_path):
        person_name_folder = os.path.join(enroll_path, person)
        embeddings = []

        for img_name in os.listdir(person_name_folder):

            img_path = os.path.join(person_name_folder, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            emb = get_embedding(img)

            if emb is None:
                continue

            embeddings.append(emb)
            if len(embeddings) == 0:
                continue

            avg_emb = np.mean(embeddings, axis=0)
        
        emb_data[person] = avg_emb
    
    return emb_data




