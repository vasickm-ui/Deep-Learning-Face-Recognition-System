import cv2
from recognize_frame import recognize_frame




def process_video(input_path, output_path, db, threshold=0.65):

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    #this could help us with debugging
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % 10 != 0:   # skip every other frame
            frame_id += 1
            continue

        results = recognize_frame(frame, db, threshold)

        #draw
        for r in results:
            box = r["bounding_box"]

            color = (0, 255, 0) if r["status"] == "known" else \
                    (0, 0, 255) if r["status"] == "unknown" else \
                    (255, 0, 0)

            cv2.rectangle(frame,
                          (box["left"], box["top"]),
                          (box["right"], box["bottom"]),
                          color, 2)

            label = r["status"]
            if r["name"]:
                label = f"{r['name']} {r['similarity_score']:.2f}"

            cv2.putText(frame,
                        label,
                        (box["left"], box["top"] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2)

        out.write(frame)

        cv2.imshow("Video Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_id += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()