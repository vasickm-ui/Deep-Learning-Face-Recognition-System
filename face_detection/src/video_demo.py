from process_video_data import process_video
from embeddings import load_db, calculate_avg_embeddings


if __name__ == "__main__":

    db_enroll = load_db("data/vectors/enroll_embeddings.pkl")
    db = calculate_avg_embeddings(db_enroll, 5)

    process_video(
        input_path="data/video/video1.mp4",
        output_path="results/tennis_annotated.mp4",
        db=db, 
        frame_rate=7
    )