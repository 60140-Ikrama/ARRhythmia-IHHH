import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from agents.base_agent import BaseAgent

class CardiacCycleAgent(BaseAgent):
    def __init__(self):
        super().__init__("CardiacCycleAgent", "ES/ED Detection and Cycle Analysis")

    def bandpass_filter(self, signal: np.ndarray, low: float = 0.5, high: float = 5.0, fs: float = 50.0) -> np.ndarray:
        """Applies a Butterworth bandpass filter to the volume curve V(t)."""
        nyquist = 0.5 * fs
        low_norm = low / nyquist
        high_norm = high / nyquist
        
        # Guard against short signals for high-order filtering
        order = 2 if len(signal) > 15 else 1
        
        b, a = butter(order, [low_norm, high_norm], btype='band')
        
        # To avoid edge artifacts, pad signal if long enough, else use simple mode
        try:
            filtered = filtfilt(b, a, signal)
            # Add mean back to keep it in original volume scale range
            filtered = filtered + np.mean(signal)
            return filtered
        except Exception as e:
            self.log(f"Filtering failed: {str(e)}. Using raw volume curve.", level=30)
            return signal

    def find_cycles(self, volume_curve: np.ndarray, fs: float = 50.0):
        """
        Detects End-Diastole (ED, peaks) and End-Systole (ES, valleys) in the volume curve.
        """
        # 1. Filter signal
        filtered_v = self.bandpass_filter(volume_curve, low=0.5, high=5.0, fs=fs)
        
        # 2. Find peaks (ED frames)
        # Heart rate range is 30-300 BPM (0.5 to 5 Hz). At fs=50, a beat takes at least 10 frames (300 BPM).
        # We can use distance = 8 frames as a safe lower limit.
        ed_frames, _ = find_peaks(filtered_v, distance=8, prominence=2.0)
        
        # 3. Find valleys (ES frames)
        # Valleys are peaks on the inverted signal
        es_frames, _ = find_peaks(-filtered_v, distance=8, prominence=2.0)
        
        # Fallback if no peaks detected (e.g. signal is too small or flat)
        if len(ed_frames) < 2 or len(es_frames) < 2:
            self.log("Peak detection with standard prominence failed. Re-trying with lower prominence...", level=30)
            ed_frames, _ = find_peaks(filtered_v, distance=8, prominence=0.5)
            es_frames, _ = find_peaks(-filtered_v, distance=8, prominence=0.5)
            
        # Hard fallback to simple local sliding window
        if len(ed_frames) < 2 or len(es_frames) < 2:
            self.log("Scipy find_peaks failed. Falling back to sliding window search.", level=30)
            ed_frames = []
            es_frames = []
            for i in range(2, len(volume_curve) - 2):
                # Local maximum
                if volume_curve[i] == max(volume_curve[i-2:i+3]):
                    ed_frames.append(i)
                # Local minimum
                if volume_curve[i] == min(volume_curve[i-2:i+3]):
                    es_frames.append(i)
            ed_frames = np.array(ed_frames)
            es_frames = np.array(es_frames)
            
        return ed_frames, es_frames, filtered_v

    def execute(self, state: dict) -> dict:
        volume_curve = state["volume_curve"]
        if volume_curve is None:
            raise ValueError("SegmentationAgent must execute before CardiacCycleAgent.")
            
        fps = float(state["metadata"].get("fps", 50.0))
        
        # Detect frames
        ed_frames, es_frames, filtered_v = self.find_cycles(volume_curve, fs=fps)
        
        # Ensure we have at least 2 cycles to compute intervals
        if len(es_frames) < 2:
            raise ValueError(f"Insufficient cardiac cycles detected (need >= 2 ES frames, got {len(es_frames)}). Cannot compute intervals.")
            
        # Compute Mechanical RR intervals in seconds (from ES to ES, which act as R-peaks)
        rr_intervals_frames = np.diff(es_frames)
        rr_intervals_sec = rr_intervals_frames / fps
        
        # Calculate Heart Rate
        mean_rr_sec = np.mean(rr_intervals_sec)
        heart_rate_bpm = 60.0 / mean_rr_sec
        
        # Calculate Ejection Fraction (EF)
        # EDV = End-Diastolic Volume, ESV = End-Systolic Volume
        # We can extract the volume values at the peak/valley frames
        ed_volumes = volume_curve[ed_frames] if len(ed_frames) > 0 else [np.max(volume_curve)]
        es_volumes = volume_curve[es_frames] if len(es_frames) > 0 else [np.min(volume_curve)]
        
        edv = np.mean(ed_volumes)
        esv = np.mean(es_volumes)
        
        if edv > 0:
            ejection_fraction = ((edv - esv) / edv) * 100.0
        else:
            ejection_fraction = 0.0
            
        # Clamp EF to [0, 100]
        ejection_fraction = max(0.0, min(100.0, ejection_fraction))
        
        # Calculate Irregularity Index (CV of RR intervals)
        if mean_rr_sec > 0:
            irregularity_index = np.std(rr_intervals_sec) / mean_rr_sec
        else:
            irregularity_index = 0.0
            
        self.log(f"Cardiac cycle analysis complete: HR={heart_rate_bpm:.1f} BPM, EF={ejection_fraction:.1f}%, Irregularity={irregularity_index:.4f}")
        
        state["es_frames"] = es_frames.tolist()
        state["ed_frames"] = ed_frames.tolist()
        state["rr_intervals_sec"] = rr_intervals_sec.tolist()
        state["heart_rate_bpm"] = float(heart_rate_bpm)
        state["ejection_fraction"] = float(ejection_fraction)
        state["irregularity_index"] = float(irregularity_index)
        
        return state
