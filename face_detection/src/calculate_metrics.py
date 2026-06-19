from embeddings import cosine_similarity, calculate_avg_embeddings, get_embedding

#calculating system performance metrics for test data set
def test_data_metrics(enroll_avg, test_db, threshold):
    misidentification = 0
    false_reject = 0
    correct = 0
    n = 0

    for person, embeddings in test_db.items():

        for emb in embeddings:
            best_similarity = -1
            best_name = "Unknown"

            for name, avg_emb in enroll_avg.items():

                cos_sim = cosine_similarity(emb, avg_emb)

                if cos_sim > threshold and cos_sim > best_similarity:
                    best_similarity = cos_sim
                    best_name = name

            if best_name == person:
                correct += 1
            elif best_name == "Unknown":
                false_reject += 1
            else:
                misidentification += 1

            n += 1

    acc = round(correct / n * 100, 4)
    frr = round(false_reject / n * 100, 4)

    return {
        "acc": acc,
        "correct": correct,
        "misidentification": misidentification,
        "false_reject": false_reject,
        "total": n,
        "frr": frr
    }


def unknown_data_metrics(enroll_avg, unknown_db, threshold):
    false_accept = 0
    correct = 0
    n = 0

    for folder_name, embeddings in unknown_db.items():

        best_similarity = -1
        best_name = "Unknown"

        for name, avg_emb in enroll_avg.items():

            for emb in embeddings:
                cos_sim = cosine_similarity(emb, avg_emb)

                if cos_sim > threshold and cos_sim > best_similarity:
                    best_similarity = cos_sim
                    best_name = name

        if best_name == "Unknown":
            correct += 1
        else:
            false_accept += 1

        n += 1

    far = round(false_accept / n * 100, 4)

    return {
        "correct": correct,
        "false_accept": false_accept,
        "total": n,
        "far": far
    }