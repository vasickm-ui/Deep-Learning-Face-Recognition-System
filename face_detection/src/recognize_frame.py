from embeddings import cosine_similarity, quality_score
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l")  # pretrained InsightFace model
app.prepare(ctx_id=0)

def recognize_frame(frame, db, threshold=0.65):
    results = []

    faces = app.get(frame)

    for face in faces:

        # default
        status = "unknown"
        name = None
        best_score = -1

        emb = face.embedding
        x1, y1, x2, y2 = face.bbox.astype(int)

        # matching
        for person, db_emb in db.items():
            score = cosine_similarity(emb, db_emb)

            if score > best_score:
                best_score = score
                name = person

        if best_score >= threshold:
            status = "known"
        else:
            status = "unknown"
            name = None

        results.append({
            "bounding_box": {"left": x1, "top": y1, "right": x2, "bottom": y2},
            "status": status,
            "name": name,
            "similarity_score": float(best_score)
        })

    return results