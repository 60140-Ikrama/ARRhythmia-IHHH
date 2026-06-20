import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from agents.base_agent import BaseAgent

try:
    import cv2
except ImportError:
    cv2 = None

from scipy.signal import savgol_filter, find_peaks


class MotionAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__("MotionAnalysisAgent", "Myocardial Motion Tracker")

    def compute_optical_flow(self, frames: np.ndarray, masks: np.ndarray) -> tuple:
        """
        Computes dense optical flow (Farneback) on LV ROI only.

        Returns:
            mean_magnitudes: (T-1,) array
            regional_flows: dict of regional curves
        """

        T = len(frames)

        mean_magnitudes = []

        septal_mags = []
        lateral_mags = []
        apical_mags = []
        basal_mags = []

        for t in range(T - 1):

            try:
                prev_f = (frames[t] * 255).astype(np.uint8)
                next_f = (frames[t + 1] * 255).astype(np.uint8)

                mask = masks[t].astype(np.uint8)

                ys, xs = np.where(mask > 0)

                if len(xs) == 0 or len(ys) == 0:
                    continue

                # ROI Bounding Box
                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()

                pad = 10

                x1 = max(0, x1 - pad)
                x2 = min(mask.shape[1], x2 + pad)

                y1 = max(0, y1 - pad)
                y2 = min(mask.shape[0], y2 + pad)

                prev_roi = prev_f[y1:y2, x1:x2]
                next_roi = next_f[y1:y2, x1:x2]
                mask_roi = mask[y1:y2, x1:x2]

                # Improved Farneback Parameters
                flow = cv2.calcOpticalFlowFarneback(
                    prev_roi,
                    next_roi,
                    None,
                    pyr_scale=0.5,
                    levels=5,
                    winsize=25,
                    iterations=5,
                    poly_n=7,
                    poly_sigma=1.5,
                    flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
                )

                flow_x = flow[..., 0]
                flow_y = flow[..., 1]

                magnitude = np.sqrt(flow_x**2 + flow_y**2)

                # Apply mask
                masked_mag = magnitude * mask_roi

                if np.sum(mask_roi) > 0:
                    mean_mag = np.sum(masked_mag) / np.sum(mask_roi)
                else:
                    mean_mag = 0.0

                mean_magnitudes.append(mean_mag)

                # Regional segmentation
                y_idx, x_idx = np.where(mask_roi > 0)

                if len(y_idx) > 0:

                    # Robust centroid
                    cx = int(np.median(x_idx))
                    cy = int(np.median(y_idx))

                    h, w = mask_roi.shape

                    y_grid, x_grid = np.ogrid[:h, :w]

                    septal_mask = (mask_roi > 0) & (x_grid < cx)
                    lateral_mask = (mask_roi > 0) & (x_grid >= cx)

                    apical_mask = (mask_roi > 0) & (y_grid < cy)
                    basal_mask = (mask_roi > 0) & (y_grid >= cy)

                    def get_mean_mag(r_mask):
                        if np.sum(r_mask) == 0:
                            return 0.0
                        return np.sum(magnitude * r_mask) / np.sum(r_mask)

                    septal_mags.append(get_mean_mag(septal_mask))
                    lateral_mags.append(get_mean_mag(lateral_mask))

                    apical_mags.append(get_mean_mag(apical_mask))
                    basal_mags.append(get_mean_mag(basal_mask))

                else:
                    septal_mags.append(0.0)
                    lateral_mags.append(0.0)
                    apical_mags.append(0.0)
                    basal_mags.append(0.0)

            except Exception as e:
                self.log(f"Optical flow failed at frame {t}: {e}")

        # Convert to arrays
        mean_magnitudes = np.array(mean_magnitudes, dtype=np.float32)

        septal_arr = np.array(septal_mags, dtype=np.float32)
        lateral_arr = np.array(lateral_mags, dtype=np.float32)
        apical_arr = np.array(apical_mags, dtype=np.float32)
        basal_arr = np.array(basal_mags, dtype=np.float32)

        # Temporal smoothing
        if len(mean_magnitudes) >= 7:

            mean_magnitudes = savgol_filter(
                mean_magnitudes,
                window_length=7,
                polyorder=2
            )

            septal_arr = savgol_filter(septal_arr, 7, 2)
            lateral_arr = savgol_filter(lateral_arr, 7, 2)
            apical_arr = savgol_filter(apical_arr, 7, 2)
            basal_arr = savgol_filter(basal_arr, 7, 2)

        regional_flows = {
            "septal": septal_arr,
            "lateral": lateral_arr,
            "apical": apical_arr,
            "basal": basal_arr
        }

        return mean_magnitudes, regional_flows

    def calculate_dyssynchrony(
        self,
        septal: np.ndarray,
        lateral: np.ndarray,
        fps: float
    ) -> float:

        if len(septal) < 10 or len(lateral) < 10:
            return 0.0

        s_norm = (septal - np.mean(septal)) / (np.std(septal) + 1e-6)
        l_norm = (lateral - np.mean(lateral)) / (np.std(lateral) + 1e-6)

        corr = np.correlate(
            s_norm - np.mean(s_norm),
            l_norm - np.mean(l_norm),
            mode='full'
        )

        corr /= (
            len(s_norm) *
            np.std(s_norm) *
            np.std(l_norm) +
            1e-6
        )

        lags = np.arange(-len(septal) + 1, len(septal))

        best_lag_frames = lags[np.argmax(corr)]

        delay_ms = (best_lag_frames / fps) * 1000.0

        return float(abs(delay_ms))

    def compute_motion_rr_variability(
        self,
        motion_curve: np.ndarray,
        fps: float
    ) -> float:

        peaks, _ = find_peaks(
            motion_curve,
            distance=max(1, int(fps // 3))
        )

        if len(peaks) < 3:
            return 0.0

        rr = np.diff(peaks) / fps

        return float(np.std(rr))

    def compute_peak_sharpness(
        self,
        motion_curve: np.ndarray
    ) -> float:

        if len(motion_curve) < 3:
            return 0.0

        gradient = np.gradient(motion_curve)

        return float(np.max(np.abs(gradient)))

    def execute(self, state: dict) -> dict:

        video = state["echo_video"]
        masks = state["segmentation_masks"]

        if video is None or masks is None:
            raise ValueError(
                "Data and Segmentation agents must execute before MotionAnalysisAgent."
            )

        fps = float(state["metadata"].get("fps", 50.0))

        is_mock = state.get("is_mock", False)

        if is_mock or cv2 is None:

            self.log("Using simulated optical flow.")

            T = len(video)

            t_axis = np.arange(T - 1)

            motion_magnitude_curve = (
                2.0 +
                1.0 * np.sin(2 * np.pi * (75 / 60.0) * (t_axis / fps)) +
                np.random.normal(0, 0.1, len(t_axis))
            )

            regional_curves = {
                "septal": motion_magnitude_curve * 0.95,
                "lateral": motion_magnitude_curve * 1.05,
                "apical": motion_magnitude_curve * 1.10,
                "basal": motion_magnitude_curve * 0.90
            }

        else:

            self.log("Computing optical flow on LV ROI...")

            motion_magnitude_curve, regional_curves = \
                self.compute_optical_flow(video, masks)

        # Dyssynchrony
        dys_index = self.calculate_dyssynchrony(
            regional_curves["septal"],
            regional_curves["lateral"],
            fps
        )

        # Motion irregularity
        mean_motion = np.mean(motion_magnitude_curve)

        if mean_motion > 0:
            mot_irregularity = (
                np.std(motion_magnitude_curve) /
                mean_motion
            )
        else:
            mot_irregularity = 0.0

        # Motion RR variability
        motion_rr_std = self.compute_motion_rr_variability(
            motion_magnitude_curve,
            fps
        )

        # Peak sharpness
        peak_sharpness = self.compute_peak_sharpness(
            motion_magnitude_curve
        )

        # Motion classification
        if mot_irregularity > 0.35:

            max_to_mean = (
                np.max(motion_magnitude_curve) /
                (mean_motion + 1e-6)
            )

            if max_to_mean > 3.0:
                motion_pattern = "pvc_jerk"
            else:
                motion_pattern = "afib_chaotic"

        else:
            motion_pattern = "normal"

        # Confidence
        motion_confidence = 1.0 - min(1.0, mot_irregularity)

        regional_scores = {
            "septal": float(np.mean(regional_curves["septal"])),
            "lateral": float(np.mean(regional_curves["lateral"])),
            "apical": float(np.mean(regional_curves["apical"])),
            "basal": float(np.mean(regional_curves["basal"]))
        }

        self.log(
            f"Motion Analysis complete | "
            f"Dyssynchrony={dys_index:.1f} ms | "
            f"Irregularity={mot_irregularity:.4f} | "
            f"Pattern={motion_pattern}"
        )

        # Store outputs
        state["motion_magnitude_curve"] = \
            motion_magnitude_curve.tolist()

        state["dyssynchrony_index_ms"] = float(dys_index)

        state["motion_irregularity"] = float(mot_irregularity)

        state["motion_rr_std"] = float(motion_rr_std)

        state["peak_sharpness"] = float(peak_sharpness)

        state["motion_confidence"] = float(motion_confidence)

        state["regional_scores"] = regional_scores

        state["motion_pattern"] = motion_pattern

        return state


if __name__ == "__main__":
    print("MotionAnalysisAgent module initialized successfully.")
    agent = MotionAnalysisAgent()
    print(f"Agent Name: {agent.name}")
    print(f"Agent Role: {agent.role}")
