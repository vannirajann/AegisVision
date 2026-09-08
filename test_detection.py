from ultralytics import YOLO

print("Loading YOLO model...")

model = YOLO("detection/yolo11n.pt")

results = model.predict(
    source="detection/videos/test.mp4",
    stream=True,
    conf=0.1,
    verbose=False
)

print("YOLO unrestricted test")

count = 0
found = False

for r in results:
    count += 1

    print("Frame", count, "Boxes:", len(r.boxes))

    if len(r.boxes) > 0:
        print("Classes:", r.boxes.cls.tolist())
        print("Confidence:", r.boxes.conf.tolist())
        found = True
        break

    if count >= 10:
        break

if found:
    print("Result: DETECTIONS FOUND")
else:
    print("Result: NO DETECTIONS")