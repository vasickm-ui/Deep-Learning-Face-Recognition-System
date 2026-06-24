from process_video_data import process_camera
from embeddings import calculate_avg_embeddings, load_db
from paths import PROJECT_ROOT

VASIC_EMBEDDINGS = PROJECT_ROOT / "data" / "vectors" / "vasic_embeddings.pkl"

if __name__ == "__main__":
    db_vasic = load_db("data/vectors/vasic_embeddings.pkl")
    db = calculate_avg_embeddings(db_vasic, 6)
    process_camera(db, "results/camera.mp4", 3)