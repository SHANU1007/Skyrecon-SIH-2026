"""
GPS + IMU Sensor Fusion — corrects Visual Odometry drift with real-world data
--------------------------------------------------------------------------------
visual_odometry.py akela sirf DIRECTION deta hai, real distance (scale) nahi,
aur chhoti-chhoti errors accumulate hoke drift ban jaati hain over time.

Yeh module VO ke output ko GPS fixes se fuse karta hai:
    1. GPS lat/lon ko local flat-ground meters (x, y) mein convert karta hai
    2. VO ke movement aur GPS ke actual movement compare karke SCALE
       estimate karta hai (kitne meters = 1 VO unit)
    3. Har GPS fix aane par fused position ko GPS ki taraf thoda khींचता
       hai (complementary filter) — isse VO drift correct hoti rehti hai
    4. IMU heading (yaw) ko bhi isi tarah VO ke estimated heading ke
       saath blend kar sakte ho drift-free orientation ke liye

Jetson pe deployment: GPS module (u-blox jaisa) generally NMEA sentences
serial port pe bhejta hai — 'parse_gga()' unhe seedha parse kar deta hai.
IMU (MPU6050/BNO055 jaisa) apna khud ka yaw/heading value deta hai apne
driver library se, jo seedha fuse_heading() mein pass kar sakte ho.
"""

import math


# ---------------------------------------------------------------------------
# GPS coordinate conversion (lat/lon -> local flat-ground meters)
# ---------------------------------------------------------------------------
EARTH_RADIUS_M = 6378137.0


def latlon_to_xy(lat, lon, origin_lat, origin_lon):
    """
    Converts GPS (lat, lon) to local (x=east, y=north) meters relative to
    an origin point. Uses an equirectangular approximation — accurate
    enough for local-area drone operations (a few km radius).
    """
    dlat = math.radians(lat - origin_lat)
    dlon = math.radians(lon - origin_lon)
    x = dlon * EARTH_RADIUS_M * math.cos(math.radians(origin_lat))  # east
    y = dlat * EARTH_RADIUS_M                                        # north
    return x, y


def parse_gga(nmea_sentence):
    """
    Parses a $GPGGA / $GNGGA NMEA sentence (standard GPS fix message) and
    returns (lat, lon, altitude_m), or None if the sentence has no valid fix.
    This is what you'll read directly off a GPS module's serial UART on Jetson.
    """
    if not (nmea_sentence.startswith("$GPGGA") or nmea_sentence.startswith("$GNGGA")):
        return None

    parts = nmea_sentence.strip().split(",")
    if len(parts) < 10 or not parts[2] or not parts[4]:
        return None

    def nmea_to_deg(raw, direction):
        deg_len = 2 if direction in ("N", "S") else 3
        deg = float(raw[:deg_len])
        minutes = float(raw[deg_len:])
        value = deg + minutes / 60.0
        return -value if direction in ("S", "W") else value

    lat = nmea_to_deg(parts[2], parts[3])
    lon = nmea_to_deg(parts[4], parts[5])
    alt = float(parts[9]) if parts[9] else 0.0
    return lat, lon, alt


# ---------------------------------------------------------------------------
# Heading fusion (VO orientation drift correction using IMU)
# ---------------------------------------------------------------------------
def fuse_heading(vo_heading_deg, imu_heading_deg, alpha=0.5):
    """
    Blends VO's estimated heading with the IMU's heading reading.
    alpha=0 -> trust VO only, alpha=1 -> trust IMU only.
    Handles 0/360-degree wraparound correctly.
    """
    diff = (imu_heading_deg - vo_heading_deg + 180) % 360 - 180
    return (vo_heading_deg + alpha * diff) % 360


# ---------------------------------------------------------------------------
# Position fusion (VO scale + drift correction using GPS)
# ---------------------------------------------------------------------------
class GPSIMUFusion:
    def __init__(self, origin_lat, origin_lon, gps_trust=0.4):
        """
        origin_lat/lon: reference GPS point (e.g. drone's takeoff location),
            all positions are reported relative to this point.
        gps_trust: how strongly GPS corrects the fused position each time a
            fix arrives (0 = ignore GPS entirely, 1 = snap fully to GPS).
        """
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.gps_trust = gps_trust

        self.scale = None          # meters per 1 VO unit, learned over time
        self._last_gps_xy = None
        self._last_vo_xy = None

        self.fused_trajectory = []  # list of (x, y) in meters

    def _update_scale_estimate(self, vo_xy, gps_xy):
        if self._last_gps_xy is None:
            self._last_gps_xy = gps_xy
            self._last_vo_xy = vo_xy
            return

        gps_dist = math.hypot(gps_xy[0] - self._last_gps_xy[0], gps_xy[1] - self._last_gps_xy[1])
        vo_dist = math.hypot(vo_xy[0] - self._last_vo_xy[0], vo_xy[1] - self._last_vo_xy[1])

        # only update scale on meaningful movement — ignore GPS jitter while hovering
        if vo_dist > 1e-3 and gps_dist > 0.5:
            new_scale = gps_dist / vo_dist
            self.scale = new_scale if self.scale is None else 0.8 * self.scale + 0.2 * new_scale

        self._last_gps_xy = gps_xy
        self._last_vo_xy = vo_xy

    def update(self, vo_xy, gps_latlon=None):
        """
        Call once per frame/timestep.
        vo_xy: (x, y) ground-plane position from VO's trajectory, unscaled.
        gps_latlon: (lat, lon) if a fresh GPS fix is available this step,
            else None (GPS fixes usually arrive slower than camera frames).

        Returns the fused (x, y) position estimate, in meters, relative to origin.
        """
        gps_xy = None
        if gps_latlon is not None:
            gps_xy = latlon_to_xy(gps_latlon[0], gps_latlon[1], self.origin_lat, self.origin_lon)
            self._update_scale_estimate(vo_xy, gps_xy)

        scale = self.scale if self.scale is not None else 1.0
        vo_scaled = (vo_xy[0] * scale, vo_xy[1] * scale)

        if gps_xy is not None:
            a = self.gps_trust
            fused = (
                (1 - a) * vo_scaled[0] + a * gps_xy[0],
                (1 - a) * vo_scaled[1] + a * gps_xy[1],
            )
        else:
            fused = vo_scaled

        self.fused_trajectory.append(fused)
        return fused

    def get_trajectory(self):
        return self.fused_trajectory


# ---------------------------------------------------------------------------
# Demo — ties visual_odometry.py together with simulated GPS fixes
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import glob
    import math
    import random
    import cv2

    from visual_odometry import MonocularVO

    parser = argparse.ArgumentParser(description="Demo: VO + GPS/IMU fusion on a folder of frames")
    parser.add_argument("--frames_dir", required=True, help="Folder of sequential drone frames")
    args = parser.parse_args()

    frame_paths = sorted(glob.glob(f"{args.frames_dir}/*"))
    if not frame_paths:
        print(f"No frames found in {args.frames_dir}")
        raise SystemExit(1)

    origin_lat, origin_lon = 24.8333, 92.7789  # example: Silchar, Assam area

    print("NOTE: this demo SIMULATES GPS fixes (no real GPS log was provided).")
    print("On the Jetson, replace the simulated `gps_latlon` below with a real")
    print("fix from parse_gga() reading your GPS module's serial NMEA output.\n")

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

    # --- Pass 2: simulate GPS fixes that follow the SAME path VO actually took
    # (assume some true-world scale + small realistic GPS noise), then fuse.
    # A real deployment skips this simulation and reads live GPS fixes instead.
    random.seed(0)
    assumed_true_scale = 3.5  # meters per VO unit, for this demo only
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(origin_lat))

    fusion = GPSIMUFusion(origin_lat, origin_lon, gps_trust=0.4)

    for i, (vx, vy) in enumerate(raw_vo_xy):
        true_east = vx * assumed_true_scale + random.gauss(0, 0.3)
        true_north = vy * assumed_true_scale + random.gauss(0, 0.3)
        sim_lat = origin_lat + true_north / meters_per_deg_lat
        sim_lon = origin_lon + true_east / meters_per_deg_lon
        gps_fix = (sim_lat, sim_lon) if i % 3 == 0 else None  # GPS updates slower than camera

        fused_xy = fusion.update((vx, vy), gps_latlon=gps_fix)
        print(f"frame {i}: VO=({vx:.2f},{vy:.2f}) unscaled  ->  fused=({fused_xy[0]:.2f},{fused_xy[1]:.2f}) m")

    print(f"\nEstimated scale (meters per VO unit): {fusion.scale}")
    print(f"Final fused position: {fusion.get_trajectory()[-1]} meters from origin")
