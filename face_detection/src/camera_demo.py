from process_video_data import process_camera
from embeddings import calculate_avg_embeddings, load_db

if __name__ == "__main__":
    db_enroll = load_db("data/vectors/enroll_embeddings.pkl")
    db = calculate_avg_embeddings(db_enroll, 5)
    process_camera(db=db)