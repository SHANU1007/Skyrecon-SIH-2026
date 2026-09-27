# SLAM — Drone Self-Localization & Mapping

Lets the drone estimate its own position/path using its camera, corrected
with GPS and IMU — useful for logging exactly where survivors/hazards were
found, and for navigating when GPS signal is weak or momentarily lost
(common near buildings/rubble in disaster zones).

## What's actually here (be honest about scope)

Full production SLAM (like ORB-SLAM3) includes loop closure, bundle
adjustment, and dense 3D mapping — that's a massive, mature codebase built
over years by robotics research groups. Reimplementing that from scratch for
a hackathon would be unrealistic and unreliable.

What's provided here is the part that matters most for this project and is
genuinely functional:

1. **`visual_odometry.py`** — Monocular Visual Odometry (VO). Tracks ORB
   features frame-to-frame and estimates the camera's relative motion
   (rotation + translation) using the essential matrix. This is the core
   "how did the camera move" building block of SLAM. **Tested on real frames
   extracted from `dashboard/CAM01.mp4`** — works out of the box.

2. **`sensor_fusion.py`** — GPS + IMU fusion. Monocular VO alone can't tell
   real-world distance (scale ambiguity) and drifts over time. This module:
   - Converts GPS lat/lon fixes to local flat-ground meters
   - Learns the real-world scale by comparing VO movement to GPS movement
   - Blends VO position with GPS fixes to correct drift (complementary filter)
   - Includes `parse_gga()` to parse real NMEA GPS sentences (what a GPS
     module actually outputs over serial on the Jetson) and `fuse_heading()`
     to blend VO's estimated heading with IMU yaw

3. **`visualize_slam.py`** — Runs both of the above and saves
   `slam_trajectory.png` comparing the raw (unscaled, drifting) VO path
   against the GPS-corrected fused path in real meters.

This is a working Visual-Inertial Odometry + Mapping pipeline — a real and
honest first version of "SLAM" for this project, not a toy demo.

## Runs on the Jetson, same as the other modules

Like `drone-ai-project/` (YOLO) and `hazard-detection/`, this is meant to run
**on the Jetson onboard the drone**, not on the ground station:

- Camera feed → straight into `visual_odometry.py` (no need to stream raw
  video off the drone — expensive on bandwidth and adds latency)
- GPS module (e.g. u-blox NEO-6M/M8N) → connected via UART/serial to the
  Jetson → its NMEA `$GPGGA` sentences get read with `pyserial` and parsed
  with `parse_gga()`
- IMU (e.g. MPU6050/BNO055) → connected via I2C/UART → its yaw reading feeds
  `fuse_heading()`
- Only the final result (drone's estimated path, in meters) needs to be sent
  over telemetry/WiFi to the ground station `dashboard/` — small, cheap data,
  unlike streaming raw video.

If you later add a LIDAR (e.g. RPLidar), that's a different SLAM approach
(point-cloud scan matching, e.g. ICP) rather than camera-based VO — different
code, but same idea: it would also run on the Jetson.

## Run it

```bash
pip install -r requirements.txt

# Visual odometry alone, on a video or webcam:
python visual_odometry.py --source path/to/video.mp4
python visual_odometry.py --source 0            # webcam

# Or on a folder of extracted frames:
python visual_odometry.py --frames_dir path/to/frames/

# Full VO + simulated GPS/IMU fusion demo (until you have a real GPS log):
python sensor_fusion.py --frames_dir path/to/frames/

# Save a before/after comparison image:
python visualize_slam.py --frames_dir path/to/frames/
```

> The fusion demo **simulates** GPS fixes since no real GPS log was provided.
> On the Jetson with a real GPS module, replace the simulated block in
> `sensor_fusion.py`'s demo with real fixes from `parse_gga()` reading your
> serial port (e.g. via `pyserial`), and real IMU yaw values from your IMU's
> driver library.

## Known limitations (say these upfront if judges ask)

- **Monocular scale ambiguity**: without GPS, VO alone doesn't know real
  distances — this is standard for single-camera systems, hence the fusion
  step. A stereo camera or depth sensor would remove this limitation.
- **No loop closure**: if the drone revisits the same spot, this won't
  recognize it and correct accumulated drift the way full SLAM
  (ORB-SLAM3/RTAB-Map) does. For a single-pass search flight this matters
  less; for long, looping missions it would.
- **Camera mount orientation matters**: the code assumes a roughly
  forward/downward-facing camera; which two axes represent the "ground
  plane" in the VO output depends on how your camera is actually mounted —
  check this against your hardware.
- For a production-grade upgrade later, look at **ORB-SLAM3** (monocular/
  stereo/RGB-D, has loop closure) or **RTAB-Map** — both are open-source and
  well-tested rather than something to rebuild from scratch.
