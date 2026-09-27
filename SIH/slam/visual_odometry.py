"""
Monocular Visual Odometry — the "camera motion tracking" core of SLAM
-----------------------------------------------------------------------
Har naye frame mein pichले frame se ORB features match karke camera
(drone) ka relative rotation + translation estimate karta hai, aur
inhe accumulate karke ek trajectory (path drone ne udaan mein liya)
banata hai.

IMPORTANT — monocular scale ambiguity:
    Ek akela camera (bina stereo/depth sensor ke) sirf motion ki
    DIRECTION bata sakta hai, asli DISTANCE (scale) nahi — matlab VO
    khud se nahi bata sakta ki drone 1 meter aage gaya ya 100 meter.
    Isliye is module ka output GPS ke saath fuse karna zaroori hai
    (dekho sensor_fusion.py) taaki real-world scale mil sake.

Yeh production-grade SLAM (ORB-SLAM3 jaisa loop-closure + bundle
adjustment wala) nahi hai — woh bahut bada undertaking hai. Yeh ek
working, honest Visual Odometry hai jo real camera frames pe chalta
hai aur real trajectory deta hai — SLAM ka pehla aur sabse zaroori
building block.
"""

import cv2
import numpy as np


class MonocularVO:
    def __init__(self, camera_matrix=None, frame_shape=None):
        """
        camera_matrix: 3x3 intrinsic matrix K. If you don't have your
            drone camera's actual calibration, pass frame_shape=(h, w)
            instead and a reasonable default K will be estimated
            (assumes ~70-degree horizontal FOV — replace with real
            calibration for accurate results).
        """
        if camera_matrix is not None:
            self.K = camera_matrix
        elif frame_shape is not None:
            h, w = frame_shape[:2]
            focal = w / (2 * np.tan(np.radians(70) / 2))
            self.K = np.array([
                [focal, 0, w / 2],
                [0, focal, h / 2],
                [0, 0, 1],
            ])
        else:
            raise ValueError("Provide either camera_matrix or frame_shape")

        self.orb = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.prev_kp = None
        self.prev_des = None

        # Accumulated pose: rotation matrix + translation vector (unscaled)
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))

        # trajectory of (x, y, z) positions in VO's own unscaled units
        self.trajectory = [(0.0, 0.0, 0.0)]

    def _match_features(self, kp1, des1, kp2, des2):
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return None, None
        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda m: m.distance)
        good = matches[: max(50, len(matches) // 2)]  # keep best half, min 50
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return pts1, pts2

    def process_frame(self, frame):
        """
        Feed one new grayscale/BGR frame. Returns True if a valid pose
        update was computed, False if this is the first frame or if
        not enough features were found to estimate motion.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        kp, des = self.orb.detectAndCompute(gray, None)

        if self.prev_kp is None:
            self.prev_kp, self.prev_des = kp, des
            return False

        pts1, pts2 = self._match_features(self.prev_kp, self.prev_des, kp, des)
        self.prev_kp, self.prev_des = kp, des

        if pts1 is None or len(pts1) < 8:
            return False

        E, mask = cv2.findEssentialMat(pts2, pts1, self.K,
                                        method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            return False

        _, R, t, mask_pose = cv2.recoverPose(E, pts2, pts1, self.K, mask=mask)

        # accumulate global pose (unscaled translation)
        self.cur_t = self.cur_t + self.cur_R @ t
        self.cur_R = R @ self.cur_R

        x, y, z = self.cur_t.flatten()
        self.trajectory.append((float(x), float(y), float(z)))
        return True

    def get_trajectory(self):
        """List of (x, y, z) positions in VO's own arbitrary (unscaled) units."""
        return self.trajectory


if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="Run monocular VO on a video or webcam")
    parser.add_argument("--source", default=0, help="0 for webcam, or path to video file")
    parser.add_argument("--frames_dir", default=None, help="Or: folder of image frames instead of video")
    args = parser.parse_args()

    vo = None

    if args.frames_dir:
        frame_paths = sorted(glob.glob(f"{args.frames_dir}/*"))
        for i, path in enumerate(frame_paths):
            frame = cv2.imread(path)
            if frame is None:
                continue
            if vo is None:
                vo = MonocularVO(frame_shape=frame.shape)
            ok = vo.process_frame(frame)
            print(f"frame {i}: {'pose updated' if ok else 'skipped (first frame / low features)'}")
    else:
        source = int(args.source) if str(args.source).isdigit() else args.source
        cap = cv2.VideoCapture(source)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if vo is None:
                vo = MonocularVO(frame_shape=frame.shape)
            vo.process_frame(frame)
        cap.release()

    if vo:
        traj = vo.get_trajectory()
        print(f"\nTotal trajectory points: {len(traj)}")
        print(f"Last position (unscaled units): {traj[-1]}")
