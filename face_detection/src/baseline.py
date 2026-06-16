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

def get_bbox(img_path):
    image = cv2.imread(img_path)

    if image is None:
        print(f"Image {img_path} not found!")

    face = app.get(image)

    if len(face) == 0:
        print(f"Image {img_path} has no faces!")

    return face[0].bbox

def cos_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

embeddings = {}
for name, path in images.items():
    emb = get_embedding(path)
    if emb is not None:
        embeddings[name] = emb

test_img = cv2.imread("test/federer_test.jpg")
test_embedding = get_embedding("test/federer_test.jpg")
test_bbox = get_bbox("test/federer_test.jpg")
if test_embedding is None or test_bbox is None:
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

#we map test bbox to integers so that we could draw rectangle and write name
x1, y1, x2, y2 = map(int, test_bbox)
color = (0, 255, 0) if best_score >= treshold else (0, 0, 255)

#drawing rectangle and writing text
cv2.rectangle(test_img, (x1, y1), (x2, y2), color, 2)
cv2.putText(
    test_img,
    best_name,
    (x1, y1-10),
    cv2.FONT_HERSHEY_DUPLEX,
    0.8,
    color,
    2
)

cv2.imshow("Result", test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


