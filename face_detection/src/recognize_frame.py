from embeddings import cosine_similarity, quality_score
from insightface.app import FaceAnalysis


def recognize_frame(frame, db, threshold, app):
    results = []

    faces = app.get(frame)

    for face in faces:

        x1, y1, x2, y2 = face.bbox.astype(int)
        crop = frame[y1:y2, x1:x2]

        # default
        status = "unknown"
        name = None
        best_score = -1

        # quality check
        q = quality_score(crop)

        if q < 0.25:
            results.append({
                "bounding_box": {"left": x1, "top": y1, "right": x2, "bottom": y2},
                "status": "low_quality",
                "name": None,
                "similarity_score": 0.0
            })
            continue

        emb = face.embedding

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