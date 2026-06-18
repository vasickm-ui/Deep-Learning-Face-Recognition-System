import numpy as np
import cv2
import csv
from insightface.app import FaceAnalysis
import os
import pandas as pd

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

    acc = round(correct/n, 4)*100
    frr = round(false_reject/n,4)*100
    return {
        "acc":acc,
        "correct": correct,
        "misidentification": misidentification,
        "false_reject": false_reject,
        "total": n,
        "frr":frr
    }

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

    far = round((false_accept / n)*100, 4)
    return {
        "correct": correct,
        "false_accept": false_accept,
        "total": n,
        "far": far
    }


def write_report(filepath, threshold, enroll_count, test_metrics, unknown_metrics):

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "a", encoding="utf-8") as f:

        f.write("CONFIGURATION\n")
        f.write("-" * 20 + "\n")
        f.write(f"Threshold: {threshold}\n")
        f.write(f"Enroll images per person: {enroll_count}\n\n")


        f.write("TEST DATASET\n")
        f.write("-" * 20 + "\n")
        f.write(f"Correct: {test_metrics['correct']}\n")
        f.write(f"Misidentification: {test_metrics['misidentification']}\n")
        f.write(f"False reject: {test_metrics['false_reject']}\n")
        f.write(f"Total: {test_metrics['total']}\n")

        if test_metrics["total"] > 0:
            acc = test_metrics["correct"] / test_metrics["total"]
        else:
            acc = 0.0

        f.write(f"Accuracy: {acc:.2%}\n\n")

        f.write("UNKNOWN DATASET\n")
        f.write("-" * 20 + "\n")
        f.write(f"Correct reject: {unknown_metrics['correct']}\n")
        f.write(f"False accept: {unknown_metrics['false_accept']}\n")
        f.write(f"Total: {unknown_metrics['total']}\n")

        if unknown_metrics["total"] > 0:
            far = unknown_metrics["false_accept"] / unknown_metrics["total"]
        else:
            far = 0.0

        f.write(f"False accept rate: {far:.2%}\n")
        f.write("\n\n\n\n")


def write_table_txt(filepath, rows):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("RESULTS TABLE\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Enroll':<8} | {'Thresh':<8} | {'Acc':<12} | {'FRR':<12} | {'FAR':<12}\n")
        f.write("-" * 65 + "\n")

        for r in rows:
            f.write(f"{r[0]:<8} | {r[1]:<8.2f} | {r[2]:<12.2f} | {r[3]:<12.2f} | {r[4]:<12.2f}\n")

table_path = "results/experiments.txt"
optimal_threshold, optimal_enroll = 0, 0
rows = []
for enroll_count in [3]:
    db = calculate_avg_embeddings("data/enroll", enroll_count)
    for threshold in [0.5, 0.65, 0.7, 0.85]:

        header = [
            "enroll_count",
            "threshold",
            "accuracy",
            "frr",
            "far"
        ]


        test_data = test_data_metrics(db, "data/test", threshold)
        unknown_data = unknown_data_metrics(db, "data/unknown", threshold)

        rows.append([
            enroll_count,
            threshold,
            test_data["acc"],
            test_data["frr"],
            unknown_data["far"]
        ])

    write_table_txt(table_path, rows)



            


            









