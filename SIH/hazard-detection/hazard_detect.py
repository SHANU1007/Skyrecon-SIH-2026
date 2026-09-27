"""
Hazard Detection — flood / fire / landslide / earthquake-debris / normal
--------------------------------------------------------------------------
Do modes mein kaam karta hai:

1) HEURISTIC MODE (default, no training needed)
   Color/texture based rules using OpenCV — turant chalega, koi dataset ya
   training ki zaroorat nahi. Accuracy trained model se kam hogi, but
   demo/prototype ke liye kaam chala dega.

2) MODEL MODE (--model hazard_best.pt)
   Agar tumne train_hazard_classifier.py se apna model train kar liya hai,
   toh usse load karke zyada accurate classification milegi.

Usage:
    python hazard_detect.py                     # webcam, heuristic mode
    python hazard_detect.py --source video.mp4  # video file, heuristic mode
    python hazard_detect.py --model hazard_best.pt   # trained model mode
"""

import argparse
import cv2
import numpy as np

CLASSES = ["flood", "fire", "landslide_debris", "earthquake_damage", "normal"]


# ---------------------------------------------------------------------------
# Heuristic mode — no training data required, pure color/texture rules
# ---------------------------------------------------------------------------
def heuristic_classify(frame):
    """Returns (label, confidence, debug_masks) using simple color-space rules."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]
    total_px = h * w

    # --- Fire / smoke: strong red-orange-yellow regions ---
    fire_lower = np.array([0, 120, 120])
    fire_upper = np.array([35, 255, 255])
    fire_mask = cv2.inRange(hsv, fire_lower, fire_upper)
    fire_ratio = cv2.countNonZero(fire_mask) / total_px

    # --- Flood / standing water: large flat blue-grey/brown regions, low texture ---
    water_lower = np.array([80, 20, 40])
    water_upper = np.array([140, 150, 220])
    water_mask = cv2.inRange(hsv, water_lower, water_upper)
    water_ratio = cv2.countNonZero(water_mask) / total_px

    # --- Debris / rubble (landslide or earthquake): high edge density, low color variety ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = cv2.countNonZero(edges) / total_px

    saturation_mean = hsv[:, :, 1].mean()

    # --- Decision rules (tune thresholds against real footage) ---
    if fire_ratio > 0.08:
        return "fire", min(0.5 + fire_ratio, 0.95), {"fire_mask": fire_mask}

    if water_ratio > 0.35:
        return "flood", min(0.5 + water_ratio * 0.5, 0.95), {"water_mask": water_mask}

    if edge_density > 0.18 and saturation_mean < 60:
        # rubble/debris tends to be low-saturation with lots of sharp edges
        # can't reliably tell landslide vs earthquake from color/texture alone —
        # flag as generic debris and recommend a trained model for the split
        return "landslide_debris", 0.45, {"edges": edges}

    return "normal", 0.6, {}


# ---------------------------------------------------------------------------
# Model mode — uses a fine-tuned classifier trained with train_hazard_classifier.py
# ---------------------------------------------------------------------------
def load_model(weights_path):
    import torch
    from torchvision import models, transforms

    checkpoint = torch.load(weights_path, map_location="cpu")
    class_names = checkpoint.get("classes", CLASSES)

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, class_names, transform


def model_classify(frame, model, class_names, transform):
    import torch

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        conf, idx = torch.max(probs, dim=0)
    return class_names[idx.item()], conf.item()


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Hazard detection (flood/fire/debris)")
    parser.add_argument("--source", default=0, help="0 for webcam, or path to video/image")
    parser.add_argument("--model", default=None, help="Path to trained hazard_best.pt (optional)")
    args = parser.parse_args()

    use_model = args.model is not None
    if use_model:
        model, class_names, transform = load_model(args.model)
        print(f"[hazard_detect] Loaded trained model, classes: {class_names}")
    else:
        print("[hazard_detect] Running in heuristic mode (no trained model given).")
        print("[hazard_detect] For better accuracy, train a model with train_hazard_classifier.py")

    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[hazard_detect] Could not open source: {source}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if use_model:
            label, conf = model_classify(frame, model, class_names, transform)
        else:
            label, conf, _ = heuristic_classify(frame)

        text = f"{label.upper()} ({conf*100:.0f}%)"
        color = (0, 0, 255) if label in ("fire", "earthquake_damage") else \
                (255, 128, 0) if label in ("flood", "landslide_debris") else (0, 200, 0)
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.imshow("Hazard Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
