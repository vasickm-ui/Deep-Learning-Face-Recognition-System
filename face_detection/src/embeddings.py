import numpy as np
import cv2
from insightface.app import FaceAnalysis
import os
import pickle


app = FaceAnalysis(name="buffalo_l")  # pretrained InsightFace model
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

#save embeddings in file so that we dont need to calculate them again 
def build_embedding_db(enroll_path, save_path):
    db = {}

    for person in os.listdir(enroll_path):
        person_folder = os.path.join(enroll_path, person)

        embeddings = []

        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            emb = get_embedding(img)

            if emb is None:
                continue

            embeddings.append(emb)

        db[person] = embeddings

    with open(save_path, "wb") as f:
        pickle.dump(db, f)

#calculating image quality
def quality_score(img):

    if img is None or img.size == 0:
        return 0.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    if h < 60 or w < 60:
        return 0.0

    if gray.std() < 18:
        return 0.0


    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(edges > 0)

    if edge_density < 0.01:
        return 0.0

    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = min(blur / 150.0, 1.0)

    brightness = np.mean(gray)
    brightness_score = 1.0 - abs(brightness - 120) / 120
    brightness_score = np.clip(brightness_score, 0.0, 1.0)

    quality = (
        0.6 * blur_score +
        0.3 * edge_density +
        0.1 * brightness_score
    )

    return quality


#this method calculates avg emedings per person and stores it in dict
def calculate_avg_embeddings(emb_data, n):
    avg_data = {}

    for person, embeddings in emb_data.items():
        if len(embeddings) == 0:
            break

        avg_data[person] = np.mean(embeddings[:n], axis=0)

    return avg_data

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

def load_db(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
if __name__ == "__main__":
    #These scripts are runed only once to create vector db
    print("Building DB")
    # print("Buliding embedding DB for Vasic")
    # build_embedding_db("data/company_persons", "data/vectors/vasic_embeddings.pkl")
    # print("Done")

    # print("Building embedding DB...")
    # build_embedding_db("data/enroll", "data/vectors/enroll_embeddings.pkl")
    # print("Done")

    # print("Building embedding DB for test data")
    # build_embedding_db("data/test", "data/vectors/test_embeddings.pkl")
    # print("Done")

    # print("Building embedding DB for unknown data")
    # build_embedding_db("data/unknown", "data/vectors/unknown_embeddings.pkl")
    # print("Done")