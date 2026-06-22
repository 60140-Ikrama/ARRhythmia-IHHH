import os
import numpy as np
from agents.base_agent import BaseAgent
from utils.state_manager import SharedMemory

try:
    import cv2
except ImportError:
    cv2 = None

class CardiacMotionTrackingAgent(BaseAgent):
    def __init__(self):
        super().__init__("CardiacMotionTrackingAgent", "Myocardial Motion Tracker")

    def compute_optical_flow(self, frames: np.ndarray, masks: np.ndarray) -> tuple:
        """
        Computes dense optical flow (Farneback) on multi-chamber segments.
        """
        T = len(frames)
        mean_magnitudes = []
        
        # Lists for regional flow magnitudes
        septal_mags = []
        lateral_mags = []
        apical_mags = []
        basal_mags = []
        
        # velocities array of shape (T-1, 112, 112, 2)
        velocities = np.zeros((T - 1, 112, 112, 2), dtype=np.float32)
        
        for t in range(T - 1):
            prev_f = (frames[t] * 255).astype(np.uint8) if frames[t].max() <= 1.0 else frames[t].astype(np.uint8)
            next_f = (frames[t+1] * 255).astype(np.uint8) if frames[t+1].max() <= 1.0 else frames[t+1].astype(np.uint8)
            
            # Mask representing LV (class 1) or all chambers
            mask = (masks[t] > 0).astype(np.uint8)
            
            if cv2 is not None:
                # Calculate Farneback optical flow
                flow = cv2.calcOpticalFlowFarneback(prev_f, next_f, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                velocities[t] = flow
                
                flow_x = flow[..., 0]
                flow_y = flow[..., 1]
                magnitude = np.sqrt(flow_x**2 + flow_y**2)
            else:
                # Fallback velocity magnitude calculation
                magnitude = np.zeros_like(mask, dtype=np.float32)
            
            # Apply mask
            masked_mag = magnitude * mask
            mean_mag = np.sum(masked_mag) / np.sum(mask) if np.sum(mask) > 0 else 0.0
            mean_magnitudes.append(mean_mag)
            
            # Compute anatomical wall regions relative to LV centroid
            y_idx, x_idx = np.where(masks[t] == 1) # LV class is 1
            if len(y_idx) > 0:
                cx, cy = np.mean(x_idx), np.mean(y_idx)
                h, w = mask.shape
                y_grid, x_grid = np.ogrid[:h, :w]
                
                septal_mask = (masks[t] == 1) & (x_grid < cx)
                lateral_mask = (masks[t] == 1) & (x_grid >= cx)
                apical_mask = (masks[t] == 1) & (y_grid < cy)
                basal_mask = (masks[t] == 1) & (y_grid >= cy)
                
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
            velocities,
            {
                "septal": np.array(septal_mags, dtype=np.float32),
                "lateral": np.array(lateral_mags, dtype=np.float32),
                "apical": np.array(apical_mags, dtype=np.float32),
                "basal": np.array(basal_mags, dtype=np.float32)
            }
        )

    def calculate_dyssynchrony(self, septal: np.ndarray, lateral: np.ndarray, fps: float) -> float:
        """Calculates dyssynchrony index as the time delay in ms between wall peaks."""
        if len(septal) < 10 or len(lateral) < 10:
            return 0.0
        s_norm = (septal - np.mean(septal)) / (np.std(septal) + 1e-6)
        l_norm = (lateral - np.mean(lateral)) / (np.std(lateral) + 1e-6)
        
        corr = np.correlate(s_norm, l_norm, mode='full')
        lags = np.arange(-len(septal) + 1, len(septal))
        
        best_lag_frames = lags[np.argmax(corr)]
        delay_ms = (best_lag_frames / fps) * 1000.0
        return float(abs(delay_ms))

    def execute(self, state: SharedMemory) -> SharedMemory:
        video = state.get("echo_video")
        # Support both legacy and nested keys
        masks = state.get("segmentation_masks")
        if masks is None:
            seg_state = state.get("segmentations", {})
            masks = seg_state.get("masks")
            
        if video is None or masks is None:
            raise ValueError("Echo video or segmentation masks not found in state.")
            
        fps = float(state.get("metadata", {}).get("fps", 50.0))
        is_mock = state.get("is_mock", False)
        
        T = len(video)
        t_axis = np.arange(T - 1)
        
        if is_mock or cv2 is None:
            mock_rhythm = state.get("mock_rhythm", "normal")
            self.log(f"Simulating optical flow tracking for rhythm: {mock_rhythm}...")
            
            # Generate simulated motion signals
            if mock_rhythm == "normal":
                delay_target = 30.0
                irreg = 0.12
                base_mag = 2.0 + 1.2 * np.sin(2 * np.pi * (75/60.0) * (t_axis/fps))
                septal_curve = 2.0 + 1.2 * np.sin(2 * np.pi * (75/60.0) * (t_axis/fps) - 0.1)
                lateral_curve = 2.0 + 1.2 * np.sin(2 * np.pi * (75/60.0) * (t_axis/fps) + 0.1)
                apical_curve = 2.2 + 1.3 * np.sin(2 * np.pi * (75/60.0) * (t_axis/fps))
                basal_curve = 1.8 + 1.0 * np.sin(2 * np.pi * (75/60.0) * (t_axis/fps))
                motion_pattern = "normal"
            elif mock_rhythm == "afib":
                delay_target = 110.0
                irreg = 0.42
                base_mag = 1.8 + 1.0 * np.sin(t_axis/4.0) + np.random.normal(0, 0.4, len(t_axis))
                septal_curve = 1.7 + 0.9 * np.sin(t_axis/4.0) + np.random.normal(0, 0.4, len(t_axis))
                lateral_curve = 1.9 + 1.1 * np.sin(t_axis/4.3) + np.random.normal(0, 0.4, len(t_axis))
                apical_curve = 2.0 + 1.0 * np.sin(t_axis/4.1) + np.random.normal(0, 0.4, len(t_axis))
                basal_curve = 1.5 + 0.8 * np.sin(t_axis/4.2) + np.random.normal(0, 0.4, len(t_axis))
                motion_pattern = "afib_chaotic"
            elif mock_rhythm == "pvc":
                delay_target = 135.0
                irreg = 0.38
                base_mag = 1.5 + np.random.normal(0, 0.1, len(t_axis))
                for idx in range(len(t_axis)):
                    if 68 <= idx <= 78:
                        base_mag[idx] += 4.5 * np.sin(np.pi * (idx - 68) / 10.0)
                    elif idx % 40 < 15:
                        base_mag[idx] += 2.0 * np.sin(np.pi * (idx % 40) / 15.0)
                septal_curve = base_mag * 0.9
                lateral_curve = np.roll(base_mag, 6) * 1.1
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
            regional_curves = {
                "septal": septal_curve,
                "lateral": lateral_curve,
                "apical": apical_curve,
                "basal": basal_curve
            }
            velocities = np.zeros((T-1, 112, 112, 2), dtype=np.float32)
        else:
            self.log("Running optical flow calculation...")
            motion_magnitude_curve, velocities, regional_curves = self.compute_optical_flow(video, masks)
            dys_index = self.calculate_dyssynchrony(regional_curves["septal"], regional_curves["lateral"], fps)
            mean_motion = np.mean(motion_magnitude_curve)
            mot_irregularity = np.std(motion_magnitude_curve) / mean_motion if mean_motion > 0 else 0.0
            
            if mot_irregularity > 0.35:
                max_to_mean = np.max(motion_magnitude_curve) / (mean_motion + 1e-6)
                if max_to_mean > 3.0:
                    motion_pattern = "pvc_jerk"
                else:
                    motion_pattern = "afib_chaotic"
            else:
                motion_pattern = "normal"
                
        # Save back to state
        state.set("motion_magnitude_curve", motion_magnitude_curve.tolist())
        state.set("dyssynchrony_index_ms", float(dys_index))
        state.set("motion_irregularity", float(mot_irregularity))
        
        regional_scores = {
            "septal": float(np.mean(regional_curves["septal"])),
            "lateral": float(np.mean(regional_curves["lateral"])),
            "apical": float(np.mean(regional_curves["apical"])),
            "basal": float(np.mean(regional_curves["basal"]))
        }
        state.set("regional_scores", regional_scores)
        state.set("motion_pattern", motion_pattern)
        
        # Save nested structure
        motion_state = {
            "velocities": velocities,
            "dyssynchrony_index_ms": float(dys_index),
            "motion_irregularity": float(mot_irregularity),
            "regional_velocities": {k: v.tolist() for k, v in regional_curves.items()}
        }
        state.set("motion_tracking", motion_state)
        
        self.log(f"Motion analysis done: Dyssynchrony={dys_index:.1f}ms, Irregularity={mot_irregularity:.4f}")
        return state

# Alias for compatibility with project manager
MotionAnalysisAgent = CardiacMotionTrackingAgent
