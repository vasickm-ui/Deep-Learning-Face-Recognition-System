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

def get_faces(img_path):
    image = cv2.imread(img_path)

    if image is None:
        print(f"Image {img_path} not found!")

    face_objects = app.get(image)
    if len(face_objects) == 0:
        print(f"Image {img_path} has no faces!")

    faces = []
    for f in face_objects:
        faces.append((f.embedding, f.bbox))

    return faces

#this calculates embeddings of known people pictures
#all of those pictures have only one face
known_people_embeddings = {}
for name, path in images.items():
    img = cv2.imread(path)
    face = app.get(img)

    if face is not None:
        known_people_embeddings[name] = face[0].embedding
    
def cos_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

test_img = cv2.imread("test/federe_murr_test.jpg")
test_faces = get_faces("test/federe_murr_test.jpg")

if test_img is None or len(test_faces) == 0:
    print("Problem with test image!")

#going through each face on picture
for emb, bbox in test_faces:
    best_name = "Unknown"
    best_score = -1
    treshold = 0.65
    for name, emb_known in known_people_embeddings.items():
        score = cos_similarity(emb, emb_known)
        print(f"{name}: {score:.4f}")

        if score > best_score and score > treshold:
            best_score = score
            best_name = name

    print(best_name)
    #we map test bbox to integers so that we could draw rectangle and write name
    x1, y1, x2, y2 = map(int, bbox)
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

    print("----------------------------------------------------------")
cv2.imwrite("results/fed_mur.jpg", test_img)
cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.imshow("Result", test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


