import numpy as np
from agents.base_agent import BaseAgent

try:
    import cv2
except ImportError:
    cv2 = None

class MotionAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__("MotionAnalysisAgent", "Myocardial Motion Tracker")

    def compute_optical_flow(self, frames: np.ndarray, masks: np.ndarray) -> tuple:
        """
        Computes dense optical flow (Farneback) on masked LV region.
        Returns:
            mean_magnitudes: (T-1,) array of mean flow magnitude per frame.
            regional_flows: dict of (T-1,) arrays for septal, lateral, apical, basal.
        """
        T = len(frames)
        mean_magnitudes = []
        
        # Lists for regional flow magnitudes
        septal_mags = []
        lateral_mags = []
        apical_mags = []
        basal_mags = []
        
        for t in range(T - 1):
            prev_f = (frames[t] * 255).astype(np.uint8)
            next_f = (frames[t+1] * 255).astype(np.uint8)
            mask = masks[t]
            
            # Calculate Farneback optical flow
            # parameters: prev_img, next_img, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
            flow = cv2.calcOpticalFlowFarneback(prev_f, next_f, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            flow_x = flow[..., 0]
            flow_y = flow[..., 1]
            magnitude = np.sqrt(flow_x**2 + flow_y**2)
            
            # Apply LV mask
            masked_mag = magnitude * mask
            mean_mag = np.sum(masked_mag) / np.sum(mask) if np.sum(mask) > 0 else 0.0
            mean_magnitudes.append(mean_mag)
            
            # Divide mask into quadrants based on centroid
            y_idx, x_idx = np.where(mask > 0)
            if len(y_idx) > 0:
                cx, cy = np.mean(x_idx), np.mean(y_idx)
                
                # Create quadrant masks
                h, w = mask.shape
                y_grid, x_grid = np.ogrid[:h, :w]
                
                # Anatomical region definition relative to centroid:
                # - Apical: y < cy (top half of sector)
                # - Basal: y >= cy (bottom half of sector)
                # - Septal: x < cx (left side)
                # - Lateral: x >= cx (right side)
                
                septal_mask = (mask > 0) & (x_grid < cx)
                lateral_mask = (mask > 0) & (x_grid >= cx)
                apical_mask = (mask > 0) & (y_grid < cy)
                basal_mask = (mask > 0) & (y_grid >= cy)
                
                def get_mean_mag(r_mask):
                    return np.sum(magnitude * r_mask) / np.sum(r_mask) if np.sum(r_mask) > 0 else 0.0
                
                septal_mags.append(get_mean_mag(septal_mask))
                lateral_mags.append(get_mean_mag(lateral_mask))
                apical_mags.append(get_mean_mag(apical_mask))
                basal_mags.append(get_mean_mag(basal_mask))
            else:
                septal_mags.append(0.0)
                lateral_mags.append(0.0)
                apical_mags.append(0.0)
                basal_mags.append(0.0)
                
        return (
            np.array(mean_magnitudes, dtype=np.float32),
            {
                "septal": np.array(septal_mags, dtype=np.float32),
                "lateral": np.array(lateral_mags, dtype=np.float32),
                "apical": np.array(apical_mags, dtype=np.float32),
                "basal": np.array(basal_mags, dtype=np.float32)
            }
        )

    def calculate_dyssynchrony(self, septal: np.ndarray, lateral: np.ndarray, fps: float) -> float:
        """
        Calculates dyssynchrony index as the time delay in milliseconds 
        between the peak motions of septal and lateral walls.
        We use cross-correlation to find the phase lag.
        """
        if len(septal) < 10 or len(lateral) < 10:
            return 0.0
            
        # Standardize signals for cross-correlation
        s_norm = (septal - np.mean(septal)) / (np.std(septal) + 1e-6)
        l_norm = (lateral - np.mean(lateral)) / (np.std(lateral) + 1e-6)
        
        # Cross correlation
        corr = np.correlate(s_norm, l_norm, mode='full')
        lags = np.arange(-len(septal) + 1, len(septal))
        
        best_lag_frames = lags[np.argmax(corr)]
        delay_ms = (best_lag_frames / fps) * 1000.0
        
        # Return absolute delay
        return float(abs(delay_ms))

    def execute(self, state: dict) -> dict:
        video = state["echo_video"]
        masks = state["segmentation_masks"]
        
        if video is None or masks is None:
            raise ValueError("Data and Segmentation agents must execute before MotionAnalysisAgent.")
            
        fps = float(state["metadata"].get("fps", 50.0))
        is_mock = state.get("is_mock", False)
        
        if is_mock or cv2 is None:
            # Simulated optical flow and regional curves
            T = len(video)
            t_axis = np.arange(T - 1)
            mock_rhythm = state.get("mock_rhythm", "normal")
            
            self.log("Using simulated optical flow and wall motion curves.")
            
            # Generate simulated motion curves with characteristic dyssynchrony and irregularities
            if mock_rhythm == "normal":
                delay_target = 30.0 # normal physiological delay
                irreg = 0.12
                # Periodic motion magnitude corresponding to heart beats
                base_mag = 2.0 + 1.2 * np.sin(2 * np.pi * (75/60.0) * (t_axis/fps))
                septal_curve = 2.0 + 1.2 * np.sin(2 * np.pi * (75/60.0) * (t_axis/fps) - 0.1)
                lateral_curve = 2.0 + 1.2 * np.sin(2 * np.pi * (75/60.0) * (t_axis/fps) + 0.1)
                apical_curve = 2.2 + 1.3 * np.sin(2 * np.pi * (75/60.0) * (t_axis/fps))
                basal_curve = 1.8 + 1.0 * np.sin(2 * np.pi * (75/60.0) * (t_axis/fps))
                motion_pattern = "normal"
            elif mock_rhythm == "afib":
                delay_target = 110.0 # high dyssynchrony
                irreg = 0.42 # high chaos
                base_mag = 1.8 + 1.0 * np.sin(t_axis/4.0) + np.random.normal(0, 0.4, len(t_axis))
                septal_curve = 1.7 + 0.9 * np.sin(t_axis/4.0) + np.random.normal(0, 0.4, len(t_axis))
                lateral_curve = 1.9 + 1.1 * np.sin(t_axis/4.3) + np.random.normal(0, 0.4, len(t_axis))
                apical_curve = 2.0 + 1.0 * np.sin(t_axis/4.1) + np.random.normal(0, 0.4, len(t_axis))
                basal_curve = 1.5 + 0.8 * np.sin(t_axis/4.2) + np.random.normal(0, 0.4, len(t_axis))
                motion_pattern = "afib_chaotic"
            elif mock_rhythm == "pvc":
                delay_target = 135.0 # very dyssynchronous ventricular beat
                irreg = 0.38
                # PVC shape: sudden jerks
                base_mag = 1.5 + np.random.normal(0, 0.1, len(t_axis))
                # Add PVC contraction peak around frame 70-80
                for idx in range(len(t_axis)):
                    if 68 <= idx <= 78:
                        base_mag[idx] += 4.5 * np.sin(np.pi * (idx - 68) / 10.0)
                    elif idx % 40 < 15:
                        base_mag[idx] += 2.0 * np.sin(np.pi * (idx % 40) / 15.0)
                
                # Make curves dyssynchronous for PVC
                septal_curve = base_mag * 0.9
                lateral_curve = np.roll(base_mag, 6) * 1.1 # 6 frames lag ~ 120ms
                apical_curve = base_mag * 1.2
                basal_curve = base_mag * 0.8
                motion_pattern = "pvc_jerk"
            elif mock_rhythm == "bradycardia":
                delay_target = 35.0
                irreg = 0.14
                base_mag = 2.0 + 1.2 * np.sin(2 * np.pi * (48/60.0) * (t_axis/fps))
                septal_curve = 2.0 + 1.2 * np.sin(2 * np.pi * (48/60.0) * (t_axis/fps) - 0.1)
                lateral_curve = 2.0 + 1.2 * np.sin(2 * np.pi * (48/60.0) * (t_axis/fps) + 0.1)
                apical_curve = 2.2 + 1.3 * np.sin(2 * np.pi * (48/60.0) * (t_axis/fps))
                basal_curve = 1.8 + 1.0 * np.sin(2 * np.pi * (48/60.0) * (t_axis/fps))
                motion_pattern = "normal"
            else: # tachycardia
                delay_target = 25.0
                irreg = 0.15
                base_mag = 1.7 + 0.9 * np.sin(2 * np.pi * (125/60.0) * (t_axis/fps))
                septal_curve = 1.7 + 0.9 * np.sin(2 * np.pi * (125/60.0) * (t_axis/fps) - 0.05)
                lateral_curve = 1.7 + 0.9 * np.sin(2 * np.pi * (125/60.0) * (t_axis/fps) + 0.05)
                apical_curve = 1.9 + 1.0 * np.sin(2 * np.pi * (125/60.0) * (t_axis/fps))
                basal_curve = 1.5 + 0.8 * np.sin(2 * np.pi * (125/60.0) * (t_axis/fps))
                motion_pattern = "normal"
                
            motion_magnitude_curve = base_mag + np.random.normal(0, 0.05, len(t_axis))
            dys_index = delay_target + np.random.normal(0, 2.0)
            mot_irregularity = irreg + np.random.normal(0, 0.01)
            regional_scores = {
                "septal": float(np.mean(septal_curve)),
                "lateral": float(np.mean(lateral_curve)),
                "apical": float(np.mean(apical_curve)),
                "basal": float(np.mean(basal_curve))
            }
            
        else:
            self.log("Computing Farneback optical flow on video frames...")
            # Compute real Farneback optical flow
            motion_magnitude_curve, regional_curves = self.compute_optical_flow(video, masks)
            
            # Compute dyssynchrony index between septal and lateral regional curves
            dys_index = self.calculate_dyssynchrony(regional_curves["septal"], regional_curves["lateral"], fps)
            
            # Compute motion irregularity index
            mean_motion = np.mean(motion_magnitude_curve)
            mot_irregularity = np.std(motion_magnitude_curve) / mean_motion if mean_motion > 0 else 0.0
            
            # Classify motion pattern based on metrics
            if mot_irregularity > 0.35:
                # AFib is chaotic, PVC is sudden jerk. We check peak density / peak variance
                # PVC will have isolated high peaks
                max_to_mean = np.max(motion_magnitude_curve) / (mean_motion + 1e-6)
                if max_to_mean > 3.0:
                    motion_pattern = "pvc_jerk"
                else:
                    motion_pattern = "afib_chaotic"
            else:
                motion_pattern = "normal"
                
            regional_scores = {
                "septal": float(np.mean(regional_curves["septal"])),
                "lateral": float(np.mean(regional_curves["lateral"])),
                "apical": float(np.mean(regional_curves["apical"])),
                "basal": float(np.mean(regional_curves["basal"]))
            }
            
        self.log(f"Motion Analysis complete. Dyssynchrony={dys_index:.1f}ms, Irregularity={mot_irregularity:.4f}, Pattern={motion_pattern}")
        
        state["motion_magnitude_curve"] = motion_magnitude_curve.tolist()
        state["dyssynchrony_index_ms"] = float(dys_index)
        state["motion_irregularity"] = float(mot_irregularity)
        state["regional_scores"] = regional_scores
        state["motion_pattern"] = motion_pattern
        
        return state
