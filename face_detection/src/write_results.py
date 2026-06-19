import os

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


def write_table_txt(filepath, rows, threshold, enroll):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("RESULTS TABLE\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Enroll':<12} | {'Threshold':<12} | {'Acc':<12} | {'FRR':<12} | {'FAR':<12}\n")
        f.write("-" * 95 + "\n")

        for r in rows:
            f.write(f"{r[0]:<12} | {r[1]:<12.2f} | {r[2]:<12.2f} | {r[3]:<12.2f} | {r[4]:<12.2f}\n")

        f.write(f"Optimal threshold: {threshold:.2f}\n")
        f.write(f"Optimal number of enrollment images: {enroll:.2f}")