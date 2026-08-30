import cv2
from video_source import VideoSource


source = VideoSource(0)

try:
    source.open()
    print("Camera opened successfully")
    print("Press Q to close the camera")

    while True:
        ret, frame = source.read()

        if not ret or frame is None:
            print("ERROR: Could not read frame")
            break

        cv2.imshow("AegisVision Camera Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    source.release()
    cv2.destroyAllWindows()
    print("Camera released")