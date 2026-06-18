from write_results import write_table_txt
from embeddings import calculate_avg_embeddings
from calculate_metrics import test_data_metrics, unknown_data_metrics

table_path = "results/experiments_main.txt"
optimal_threshold, optimal_enroll, score = 0, 0, 0
rows = []
score, max_score = 0, 0
for enroll_count in [1, 3, 5]:

    db = calculate_avg_embeddings("data/enroll", enroll_count)
    for threshold in [0.5, 0.65, 0.7, 0.75, 0.85]:

        header = [
            "enroll_count",
            "threshold",
            "accuracy",
            "frr",
            "far"
        ]


        test_data = test_data_metrics(db, "data/test", threshold)
        unknown_data = unknown_data_metrics(db, "data/unknown", threshold)

        score = 1 - (test_data["frr"]/100 + 1.5*unknown_data["far"]/100)/2
        if score > max_score:
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