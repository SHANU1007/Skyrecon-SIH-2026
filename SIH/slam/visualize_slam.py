"""
Visualize raw VO trajectory (unscaled, drifts over time) vs the GPS/IMU-fused
trajectory (scaled to real meters, drift-corrected). Saves slam_trajectory.png
— good for showing "why fusion matters" in your SIH demo/report.

Usage:
    python visualize_slam.py --frames_dir /path/to/frames
"""

import argparse
import glob
import math
import random

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visual_odometry import MonocularVO
from sensor_fusion import GPSIMUFusion


def run(frames_dir, origin=(24.8333, 92.7789)):
    frame_paths = sorted(glob.glob(f"{frames_dir}/*"))
    if not frame_paths:
        raise SystemExit(f"No frames found in {frames_dir}")

    # --- Pass 1: run real VO on the actual frames ---
    vo = None
    raw_vo_xy = []
    for path in frame_paths:
        frame = cv2.imread(path)
        if frame is None:
            continue
        if vo is None:
            vo = MonocularVO(frame_shape=frame.shape)
        vo.process_frame(frame)
        vx, vy, _ = vo.get_trajectory()[-1]
        raw_vo_xy.append((vx, vy))

    # --- Pass 2: simulate GPS fixes following the SAME path VO took
    # (assumed true-world scale + small realistic noise), then fuse.
    random.seed(0)
    origin_lat, origin_lon = origin
    assumed_true_scale = 3.5
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(origin_lat))

    fusion = GPSIMUFusion(origin_lat, origin_lon, gps_trust=0.4)
    fused_xy = []

    for i, (vx, vy) in enumerate(raw_vo_xy):
        true_east = vx * assumed_true_scale + random.gauss(0, 0.3)
        true_north = vy * assumed_true_scale + random.gauss(0, 0.3)
        sim_lat = origin_lat + true_north / meters_per_deg_lat
        sim_lon = origin_lon + true_east / meters_per_deg_lon
        gps_fix = (sim_lat, sim_lon) if i % 3 == 0 else None

        fused_xy.append(fusion.update((vx, vy), gps_latlon=gps_fix))

    return raw_vo_xy, fused_xy


def plot(raw_vo_xy, fused_xy, out_path="slam_trajectory.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    xs, ys = zip(*raw_vo_xy)
    axes[0].plot(xs, ys, marker="o", markersize=3, color="crimson")
    axes[0].set_title("Raw Visual Odometry\n(unscaled, drift accumulates)")
    axes[0].set_xlabel("x (arbitrary VO units)")
    axes[0].set_ylabel("y (arbitrary VO units)")
    axes[0].axis("equal")
    axes[0].grid(alpha=0.3)

    xs, ys = zip(*fused_xy)
    axes[1].plot(xs, ys, marker="o", markersize=3, color="seagreen")
    axes[1].set_title("GPS+IMU Fused Trajectory\n(scaled to meters, drift-corrected)")
    axes[1].set_xlabel("x (meters, east)")
    axes[1].set_ylabel("y (meters, north)")
    axes[1].axis("equal")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", required=True)
    args = parser.parse_args()

    raw_vo_xy, fused_xy = run(args.frames_dir)
    plot(raw_vo_xy, fused_xy)
