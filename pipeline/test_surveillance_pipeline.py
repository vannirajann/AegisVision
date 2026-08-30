import cv2

from surveillance_pipeline import SurveillancePipeline


print("Starting AegisVision surveillance test...")

pipeline = SurveillancePipeline()

try:
    pipeline.initialize()

    print("Live AegisVision surveillance started")
    print("Press Q to quit")

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

        # Draw face boxes
        for face in faces:

            x = face["x"]
            y = face["y"]
            w = face["width"]
            h = face["height"]

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        # Face count
        cv2.putText(
            frame,
            f"Faces: {len(faces)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Person count
        cv2.putText(
            frame,
            f"People: {len(people)}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Movement
        movement_text = (
            "MOVEMENT DETECTED"
            if movement
            else "NO MOVEMENT"
        )

        cv2.putText(
            frame,
            movement_text,
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Event notification
        if event is not None:

            cv2.putText(
                frame,
                "EVENT GENERATED",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.imshow(
            "AegisVision - Surveillance Pipeline",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    pipeline.release()
    cv2.destroyAllWindows()

    print("AegisVision surveillance stopped")