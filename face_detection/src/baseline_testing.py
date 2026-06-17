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
                print(f"Cant load image with path {img_path}")
                exit()

            emb = get_embedding(img)

            if emb is None:
                continue

            embeddings.append(emb)
            if len(embeddings) == 0:
                continue

            avg_emb = np.mean(embeddings, axis=0)
        
        emb_data[person] = avg_emb
    
    return emb_data

#calculating system performance metrics for test data set
def test_data_metrics(data_dict, test_path, threshold):

    misidentification, false_reject = 0, 0
    correct, n = 0, 0
    #going through folders (djokovic, nadal, federer)
    for person in os.listdir(test_path):

        person_folder = os.path.join(test_path, person)
        #going throgh image files
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder,img_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Cant load image with path {img_path}")
                exit()

            emb = get_embedding(img)

            best_similarity = -1
            best_name = "Unknown"

            #going through all avg embeddings
            for name, avg_emb in data_dict.items():
                cos_sim = cosine_similarity(emb, avg_emb)

                if cos_sim > threshold and cos_sim > best_similarity:
                    best_name = name
                    best_similarity = cos_sim

            if best_name == person:
                correct += 1
            elif best_name == "Unknown":
                false_reject += 1
            else:
                misidentification +=1
            
            n += 1

    print(f"Correct: {correct}/{n}")
    print(f"Misidentification: {misidentification}/{n}")
    print(f"False reject: {false_reject}/{n}")

def unknown_data_metrics(data_dict, unknown_path, threshold):

    false_accept = 0
    correct, n = 0, 0

    
    #going throgh image files
    for img_name in os.listdir(unknown_path):
        img_path = os.path.join(unknown_path,img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Cant load image with path {img_path}")
            exit()

        emb = get_embedding(img)

        best_similarity = -1
        best_name = "Unknown"

        #going through all avg embeddings
        for name, avg_emb in data_dict.items():
            cos_sim = cosine_similarity(emb, avg_emb)

            if cos_sim > threshold and cos_sim > best_similarity:
                best_name = name
                best_similarity = cos_sim

        if best_name == "Unknown":
            correct += 1
        else:
            false_accept +=1
        
        n += 1

    print(f"Correct: {correct}/{n}")
    print(f"False accept: {false_accept}/{n}")

data_base = calculate_avg_embeddings("data/enroll")
unknown_data_metrics(data_base, "data/unknown", 0.65)

            


            









