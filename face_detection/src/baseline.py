import numpy as np
import cv2
from insightface.app import FaceAnalysis

#dictionary for easier image storing and accessing
images = {
    "djokovic":"data/known_people/djokovic.png",
    "alcaraz":"data/known_people/alcaraz.png",
    "federer":"data/known_people/federer.png",
    "nadal":"data/known_people/nadal.jpg",
}

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1)

#function for calculating embeddings
def get_embedding(img_path):
    image = cv2.imread(img_path)

    if image is None:
        print(f"Image {img_path} not found!")

    face = app.get(image)

    if len(face) == 0:
        print(f"Image {img_path} has no faces!")

    return face[0].embedding

def cos_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

embeddings = {}
for name, path in images.items():
    emb = get_embedding(path)
    if emb is not None:
        embeddings[name] = emb

test_embedding = get_embedding("test/test_djokovic.jpg")
if test_embedding is None:
    print("Problem with test image!")

best_name = "Unknown"
best_score = -1
treshold = 0.65
for name, emb in embeddings.items():
    score = cos_similarity(emb, test_embedding)
    print(f"{name}: {score:.4f}")

    if score > best_score and score > treshold:
        best_score = score
        best_name = name

print(best_name)


