# SIH — Search & Rescue Drone Project (PS 26177)

Drone-based disaster search & rescue system, built for Smart India Hackathon
(Problem Statement 26177). The drone finds survivors, identifies the hazard
around them, tracks its own position, and a ground team gets the safest
route to reach them.

## System Architecture

```
                         ONBOARD THE DRONE (Jetson)
   ┌─────────────────────────────────────────────────────────────────┐
   │  Camera ──► drone-ai-project/  (YOLO: where are the survivors?)  │
   │  Camera ──► hazard-detection/  (what hazard: flood/fire/debris?) │
   │  Camera+GPS+IMU ──► slam/      (where is the drone right now?)  │
   └─────────────────────────────────────────────────────────────────┘
                                   │
                    only small results sent over telemetry/WiFi
                    (not raw video — bandwidth + latency)
                                   ▼
                       GROUND STATION (laptop/base)
   ┌─────────────────────────────────────────────────────────────────┐
   │  path-planning/  → best rescue route to survivors, avoiding      │
   │                     hazards (A* / Dijkstra)                      │
   │  dashboard/      → SAR-1 ground station UI showing all of it     │
   └─────────────────────────────────────────────────────────────────┘
```

All four AI/algorithm modules (`drone-ai-project`, `hazard-detection`,
`path-planning`, `slam`) run as independent Python scripts right now —
they're not wired together yet. The integration work (having them share
data live, e.g. over a small local API/WebSocket into the dashboard) is
the natural next step and is called out in each module's own README.

## Structure

```
SIH/
├── drone-ai-project/       # YOLOv8-based human detection
│   ├── best.pt             # Custom trained model weights
│   ├── yolov8n.pt          # Base YOLOv8 nano model
│   ├── detect.py           # Live detection using base YOLOv8 model (webcam)
│   ├── test_trained_model.py   # Live detection using custom trained model (webcam)
│   └── requirements.txt
├── hazard-detection/        # Flood / fire / landslide / earthquake classification
│   ├── hazard_detect.py           # Run detection (heuristic mode or trained model)
│   ├── train_hazard_classifier.py # Train your own classifier on labeled images
│   ├── requirements.txt
│   └── README.md            # Full setup + dataset instructions
├── path-planning/           # Best rescue route to survivors (A* + Dijkstra)
│   ├── path_planner.py      # Core algorithms + multi-survivor rescue ordering
│   ├── visualize_path.py    # Saves a PNG showing the route around hazards
│   ├── requirements.txt
│   └── README.md            # Full explanation + how it connects to the rest
├── slam/                     # Drone self-localization & mapping
│   ├── visual_odometry.py   # Camera-based motion tracking (tested on CAM01.mp4)
│   ├── sensor_fusion.py     # GPS + IMU fusion to fix VO's scale/drift
│   ├── visualize_slam.py    # Saves a before/after trajectory comparison PNG
│   ├── requirements.txt
│   └── README.md            # Full explanation, Jetson deployment notes, limitations
└── dashboard/               # Ground station web dashboard
    ├── index.html           # SAR-1 ground station UI (3D map + telemetry)
    └── CAM01.mp4            # Sample drone camera feed
```

## Setup

```bash
cd SIH/drone-ai-project
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run detection with the base YOLOv8 model:
```bash
python detect.py
```

Run detection with the custom trained model (`best.pt`):
```bash
python test_trained_model.py
```

Both scripts open your webcam (`cv2.VideoCapture(0)`) and draw bounding boxes on detected humans. Press `q` to quit.

## Hazard Detection

Classifies the drone feed as flood / fire / landslide debris / earthquake
damage / normal. Comes with a no-training heuristic mode to demo immediately,
plus a training script to fine-tune your own classifier for real accuracy.
See `hazard-detection/README.md` for full setup.

## Path Planning

Once survivors and hazards are located, `path-planning/` computes the
safest/cheapest route for a rescue team to reach them, using both A* and
Dijkstra (A* is faster, both give the optimal-cost path here). Handles
multiple survivors by planning a rescue order. See
`path-planning/README.md` for full details and how to plug in real map data.

## SLAM (Self-Localization & Mapping)

The drone tracks its own flight path using its camera (Visual Odometry),
corrected with GPS and IMU data for real-world scale and drift correction.
Tested end-to-end on real frames from `dashboard/CAM01.mp4`. See
`slam/README.md` for full details, Jetson deployment notes, and known
limitations (monocular scale ambiguity, no loop closure — see the README for
what that means and when it matters).

## Dashboard

Open `dashboard/index.html` in a browser to view the ground station UI (uses Three.js and Leaflet via CDN, so an internet connection is needed).

## Notes

- `venv/` is intentionally excluded from this repo (see `.gitignore`) — install dependencies fresh using `requirements.txt`.
- Model weight files (`best.pt`, `yolov8n.pt`) and the sample video are committed directly since they're under GitHub's file size limits; consider Git LFS if they grow larger.
