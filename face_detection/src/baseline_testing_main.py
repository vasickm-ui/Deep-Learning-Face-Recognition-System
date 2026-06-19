from write_results import write_table_txt
from embeddings import calculate_avg_embeddings, load_db
from calculate_metrics import test_data_metrics, unknown_data_metrics
import numpy as np

table_path = "results/experiments_main1335.txt"
optimal_threshold, optimal_enroll, score = 0, 0, 0
rows = []
score, max_score = 0, 0
enroll_db = load_db("data/vectors/enroll_embeddings.pkl")
test_db = load_db("data/vectors/test_embeddings.pkl")
unknown_db = load_db("data/vectors/unknown_embeddings.pkl")
for k, v in test_db.items():
    print(k)
    print(type(v))
    print(np.shape(v))

for enroll_count in [1, 3, 5]:
    for threshold in [0.5, 0.65, 0.7, 0.75, 0.85]:

        #we need to calculate avg embedings for each number of enroll images
        avg_embeddings = calculate_avg_embeddings(enroll_db, enroll_count)
        header = [
            "enroll_count",
            "threshold",
            "accuracy",
            "frr",
            "far"
        ]


        test_data = test_data_metrics(avg_embeddings, test_db, threshold)
        unknown_data = unknown_data_metrics(avg_embeddings, unknown_db, threshold)

        score = 1 - (test_data["frr"]/100 + 1.5*unknown_data["far"]/100)/2
        if score > max_score:
            max_score = score
            optimal_enroll = enroll_count
            optimal_threshold = threshold

        rows.append([
            enroll_count,
            threshold,
            test_data["acc"],
            test_data["frr"],
            unknown_data["far"]
        ])

    write_table_txt(table_path, rows, optimal_threshold, optimal_enroll)