from embeddings import cosine_similarity, quality_score
from insightface.app import FaceAnalysis
from check_liveness import check_liveness

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
        face_crop = frame[y1:y2, x1:x2]

        if quality_score(face_crop) < 0.1:
            print("Bad quality!")
            results.append({
                "bounding_box": {"left": x1, "top": y1, "right": x2, "bottom": y2},
                "status": "unknown",
                "name": "bad quality",
                "similarity_score": 0.0
            })
            continue
        print("Good quality")


        # face_crop = frame[y1:y2, x1:x2]

        # if face_crop.size == 0:
        #     continue

    
        is_live, live_score = check_liveness(face_crop, threshold=0.98)

        if not is_live:
            results.append({
                "bounding_box": {"left": x1, "top": y1, "right": x2, "bottom": y2},
                "status": "spoof",
                "name": None,
                "similarity_score": 0.0
            })
            continue

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