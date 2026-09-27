# Hazard Detection Module

Classifies the disaster type visible in a drone feed: **flood**, **fire**,
**landslide debris**, **earthquake damage**, or **normal** (no hazard).

## Two ways to run it

### 1. Heuristic mode (works right now, no training data needed)
Pure OpenCV color/texture rules — good for a quick demo:
- **Fire**: detects strong red-orange-yellow regions (flame/smoke color)
- **Flood**: detects large flat blue-grey/brown water-colored regions
- **Landslide/earthquake debris**: detects high edge density + low color
  saturation (rubble texture)

```bash
pip install -r requirements.txt
python hazard_detect.py                    # webcam
python hazard_detect.py --source clip.mp4  # video file
```

This won't be very accurate — it's a rule-based baseline so the pipeline
works end-to-end before you have a trained model.

### 2. Trained model mode (recommended for the actual demo/judging)
Fine-tunes a ResNet18 on your own labeled images for much better accuracy.

**Step 1 — collect & organize images:**
```
data/
  train/
    flood/             (50+ images)
    fire/
    landslide_debris/
    earthquake_damage/
    normal/
  val/
    flood/             (10+ images per class)
    fire/
    landslide_debris/
    earthquake_damage/
    normal/
```
Good sources: frames extracted from `CAM01.mp4`, public datasets like the
Kaggle "Disaster Images Dataset" or the xBD building-damage dataset, or
images scraped/collected for each hazard type.

**Step 2 — train:**
```bash
python train_hazard_classifier.py --data ./data --epochs 10 --out hazard_best.pt
```

**Step 3 — run detection with your trained model:**
```bash
python hazard_detect.py --model hazard_best.pt
```

## Integrating with the rest of the project

- Run this alongside `drone-ai-project/test_trained_model.py` (human
  detection) — one tells you *where people are*, this tells you *what kind
  of disaster it is*.
- To feed hazard labels into the `dashboard/index.html` ground station, have
  this script POST its output (label + confidence + timestamp) to a small
  local API/WebSocket, and add a panel in the dashboard to display it. That
  integration isn't wired up yet — currently each piece runs standalone.
