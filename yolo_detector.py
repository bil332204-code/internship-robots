"""
yolo_detector.py
------------------
Runs YOLOv8 (local, no internet/API needed) on a photo and returns
a count of each detected object type.

First run will auto-download the yolov8n.pt model (~6MB, nano version --
fast and light, good for a laptop).
"""

from ultralytics import YOLO

# Loaded lazily on first use (not at import time) so that choosing a
# different menu option (e.g. Gesture Mode) never pays the cost of loading
# YOLO in the background.
_model = None


def _get_model():
    global _model
    if _model is None:
        print("📦 Loading YOLOv8 model (first run may auto-download ~6MB)...")
        _model = YOLO("yolov8n.pt")
        print("✅ YOLOv8 model ready!")
    return _model


# Only report detections above this confidence to avoid noisy false positives
CONFIDENCE_THRESHOLD = 0.5


def detect_objects(image_path: str):
    """
    Runs YOLO on the given image file.
    Returns a dict like:
        {"bottle": {"count": 2, "confidence": 0.87},
         "person": {"count": 1, "confidence": 0.92}}
    -- one entry per object type, with how many were seen and the
    highest confidence among them.
    """
    results = _get_model()(image_path, verbose=False)[0]

    counts = {}
    for box in results.boxes:
        label = results.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        if label not in counts:
            counts[label] = {"count": 0, "confidence": 0.0}
        counts[label]["count"] += 1
        counts[label]["confidence"] = max(counts[label]["confidence"], confidence)

    return counts


def detect_with_positions(image_path: str):
    """
    Like detect_objects, but also returns each object's center position
    (both horizontal and vertical, in pixels) -- needed for centering
    the camera on an object. Returns (detections_list, image_width, image_height).

    detections_list is a list of dicts:
        {"label": "bottle", "confidence": 0.87, "center_x": 142.5, "center_y": 88.0}
    """
    results = _get_model()(image_path, verbose=False)[0]
    image_height, image_width = results.orig_shape  # (height, width)

    detections = []
    for box in results.boxes:
        label = results.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        if confidence < CONFIDENCE_THRESHOLD:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        detections.append({
            "label": label, "confidence": confidence,
            "center_x": center_x, "center_y": center_y,
        })

    return detections, image_width, image_height


if __name__ == "__main__":
    # Quick manual test: python yolo_detector.py
    results = detect_objects("snapshot.jpg")
    for label, info in results.items():
        print(f"{label}: count={info['count']}, confidence={info['confidence']:.2f}")