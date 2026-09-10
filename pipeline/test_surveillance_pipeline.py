import cv2

from surveillance_pipeline import SurveillancePipeline


def safe_imshow(frame, title):
    try:
        cv2.imshow(title, frame)
        return cv2.waitKey(1) & 0xFF
    except cv2.error:
        return None


print("Starting AegisVision surveillance test...")

pipeline = SurveillancePipeline()

try:
    pipeline.initialize()

    print("Live AegisVision surveillance started")
    print("Press Q to quit when a GUI window is available; otherwise this test will stop after a few headless frames.")

    frame_count = 0
    max_headless_frames = 5
    headless_mode = False

    while True:
        ret, frame = pipeline.video_source.read()

        if not ret or frame is None:
            print("Could not read camera frame")
            break

        results = pipeline.process_frame(frame)
        faces = results["faces"]
        people = results["people"]
        movement = results["movement"]
        event = results["event"]

        for face in faces:
            x = face["x"]
            y = face["y"]
            w = face["width"]
            h = face["height"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, f"Faces: {len(faces)}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"People: {len(people)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "MOVEMENT DETECTED" if movement else "NO MOVEMENT", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if event is not None:
            cv2.putText(frame, "EVENT GENERATED", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        key = safe_imshow(frame, "AegisVision - Surveillance Pipeline")
        if key is None:
            headless_mode = True
            if frame_count >= max_headless_frames:
                print(f"Headless mode limit reached ({max_headless_frames} frames). Stopping gracefully.")
                break
            print(f"[INFO] OpenCV GUI backend unavailable in this environment. Headless frame {frame_count + 1}/{max_headless_frames}.")
        else:
            if key == ord("q"):
                print(f"Stopped after {frame_count + 1} frames.")
                break

        frame_count += 1

finally:
    pipeline.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

    print("AegisVision surveillance stopped")