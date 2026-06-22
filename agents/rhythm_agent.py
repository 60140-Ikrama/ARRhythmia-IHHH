import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import welch
from agents.base_agent import BaseAgent
from utils.state_manager import SharedMemory

class MechanicalRhythmAgent(BaseAgent):
    """
    Computes Mechanical Rhythm Variability Metrics (MRVM) derived from mechanical contractions.
    Note: These are mechanically-derived indexes inspired by electrical ECG HRV.
    """
    def __init__(self):
        super().__init__("MechanicalRhythmAgent", "Mechanical Rhythmologist")

    def compute_time_domain(self, rr_ms: np.ndarray) -> dict:
        """Computes time domain MRVM metrics."""
        sdnn = np.std(rr_ms)
        diff_rr = np.diff(rr_ms)
        rmssd = np.sqrt(np.mean(diff_rr ** 2)) if len(diff_rr) > 0 else 0.0
        
        nn50 = np.sum(np.abs(diff_rr) > 50.0) if len(diff_rr) > 0 else 0
        pnn50 = (nn50 / len(diff_rr)) * 100.0 if len(diff_rr) > 0 else 0.0
        
        return {
            "sdnn_ms": float(sdnn),
            "rmssd_ms": float(rmssd),
            "pnn50": float(pnn50)
        }

    def compute_frequency_domain(self, rr_sec: np.ndarray) -> dict:
        """Computes frequency domain MRVM power spectral densities."""
        n_beats = len(rr_sec)
        if n_beats < 4:
            return {"lf_ms2": 0.0, "hf_ms2": 0.0, "lf_hf": 1.0}
            
        timestamps = np.cumsum(rr_sec)
        fs_interp = 4.0
        t_start, t_end = timestamps[0], timestamps[-1]
        
        if t_end - t_start <= 0.5:
            return {"lf_ms2": 0.0, "hf_ms2": 0.0, "lf_hf": 1.0}
            
        t_new = np.arange(t_start, t_end, 1.0 / fs_interp)
        if len(t_new) < 4:
            return {"lf_ms2": 0.0, "hf_ms2": 0.0, "lf_hf": 1.0}
            
        try:
            f = interp1d(timestamps, rr_sec, kind='linear', fill_value="extrapolate")
            rr_interp = f(t_new)
            rr_interp_ms = (rr_interp - np.mean(rr_interp)) * 1000.0
            
            nperseg = min(len(rr_interp_ms), 64)
            freqs, psd = welch(rr_interp_ms, fs=fs_interp, nperseg=nperseg)
            
            lf_mask = (freqs >= 0.04) & (freqs < 0.15)
            hf_mask = (freqs >= 0.15) & (freqs <= 0.40)
            
            df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
            lf_power = np.sum(psd[lf_mask]) * df if np.any(lf_mask) else 0.0
            hf_power = np.sum(psd[hf_mask]) * df if np.any(hf_mask) else 0.0
            
            lf_hf = lf_power / hf_power if hf_power > 0 else 1.0
            
            return {
                "lf_ms2": float(lf_power),
                "hf_ms2": float(hf_power),
                "lf_hf": float(lf_hf)
            }
        except Exception as e:
            self.log(f"Frequency domain calculation failed: {str(e)}", level=30)
            return {"lf_ms2": 0.0, "hf_ms2": 0.0, "lf_hf": 1.0}

    def compute_poincare(self, rr_ms: np.ndarray) -> dict:
        """Computes Poincaré features (SD1, SD2, ratio)."""
        if len(rr_ms) < 3:
            return {"sd1": 0.0, "sd2": 0.0, "sd_ratio": 1.0}
            
        diff_rr = np.diff(rr_ms)
        sd1 = np.sqrt(0.5 * np.var(diff_rr, ddof=1))
        var_rr = np.var(rr_ms, ddof=1)
        sd2_sq = 2.0 * var_rr - sd1**2
        sd2 = np.sqrt(max(0.0, sd2_sq))
        sd_ratio = sd1 / sd2 if sd2 > 0 else 1.0
        
        return {
            "sd1": float(sd1),
            "sd2": float(sd2),
            "sd_ratio": float(sd_ratio)
        }

    def compute_entropy(self, rr_ms: np.ndarray) -> float:
        """Computes Shannon-like entropy of the interval distributions."""
        if len(rr_ms) < 5:
            return 0.0
        # Compute histogram of rr intervals and calculate entropy
        hist, bin_edges = np.histogram(rr_ms, bins=5, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        return float(entropy)

    def execute(self, state: SharedMemory) -> SharedMemory:
        # Check if cardiac cycle agent has run
        rr_sec = state.get("rr_intervals_sec")
        if not rr_sec:
            # Reconstruct from ES frames if needed
            es_frames = state.get("es_frames")
            if not es_frames:
                # We extract from volumes
                # For compatibility, we can search peaks/valleys in LV volume
                lv_volumes = state.get_nested("segmentations", "volumes", {}).get("LV", [])
                if not lv_volumes:
                    lv_volumes = state.get("volume_curve", [])
                
                # Simple threshold peak detector for intervals
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(-np.array(lv_volumes), distance=8, prominence=1.0)
                es_frames = peaks.tolist()
                state.set("es_frames", es_frames)
            
            fps = float(state.get_nested("metadata", "fps", 50.0) or 50.0)
            rr_sec = (np.diff(es_frames) / fps).tolist()
            state.set("rr_intervals_sec", rr_sec)
            
        if not rr_sec:
            raise ValueError("No cardiac cycle intervals (rr_intervals_sec) found or could be computed.")
            
        rr_sec_np = np.array(rr_sec)
        rr_ms_np = rr_sec_np * 1000.0
        
        time_metrics = self.compute_time_domain(rr_ms_np)
        freq_metrics = self.compute_frequency_domain(rr_sec_np)
        poincare_metrics = self.compute_poincare(rr_ms_np)
        entropy = self.compute_entropy(rr_ms_np)
        
        # Determine classification pattern (used by legacy rules)
        sdnn = time_metrics["sdnn_ms"]
        sd_ratio = poincare_metrics["sd_ratio"]
        irreg = state.get("irregularity_index", 0.0)
        
        if irreg > 0.15 and sdnn > 80.0 and sd_ratio > 0.4:
            pattern = "afib_pattern"
        elif time_metrics["rmssd_ms"] > 60.0 and irreg > 0.08:
            pattern = "pvc_pattern"
        else:
            pattern = "normal"
            
        # Update legacy keys for compatibility
        state.set("hrv_time", time_metrics)
        state.set("hrv_freq", freq_metrics)
        state.set("hrv_nonlinear", poincare_metrics)
        state.set("hrv_pattern", pattern)
        
        # Store in state
        mrvm_state = {
            "rr_intervals_sec": rr_sec,
            "heart_rate_bpm": float(state.get("heart_rate_bpm", 75.0)),
            "irregularity_index": float(irreg),
            "sdnn_ms": time_metrics["sdnn_ms"],
            "rmssd_ms": time_metrics["rmssd_ms"],
            "pnn50": time_metrics["pnn50"],
            "lf_power": freq_metrics["lf_ms2"],
            "hf_power": freq_metrics["hf_ms2"],
            "lf_hf_ratio": freq_metrics["lf_hf"],
            "sd1": poincare_metrics["sd1"],
            "sd2": poincare_metrics["sd2"],
            "sd_ratio": poincare_metrics["sd_ratio"],
            "entropy": entropy
        }
        state.set("mrvm_features", mrvm_state)
        
        self.log(f"Computed MRVM features. Mechanical SDNN: {time_metrics['sdnn_ms']:.1f}ms, Entropy: {entropy:.2f}")
        return state
