import cv2
import os
from embeddings import cosine_similarity, calculate_avg_embeddings, get_embedding

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