import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import welch
from agents.base_agent import BaseAgent

class MechanicalHRVAgent(BaseAgent):
    def __init__(self):
        super().__init__("MechanicalHRVAgent", "Mechanical HRV Analysis")

    def compute_time_domain(self, rr_ms: np.ndarray) -> dict:
        """Computes time domain HRV metrics."""
        sdnn = np.std(rr_ms)
        diff_rr = np.diff(rr_ms)
        rmssd = np.sqrt(np.mean(diff_rr ** 2)) if len(diff_rr) > 0 else 0.0
        
        # pNN50: percentage of successive RR differences > 50ms
        nn50 = np.sum(np.abs(diff_rr) > 50.0) if len(diff_rr) > 0 else 0
        pnn50 = (nn50 / len(diff_rr)) * 100.0 if len(diff_rr) > 0 else 0.0
        
        return {
            "sdnn_ms": float(sdnn),
            "rmssd_ms": float(rmssd),
            "pnn50": float(pnn50)
        }

    def compute_frequency_domain(self, rr_sec: np.ndarray) -> dict:
        """
        Computes frequency domain HRV metrics using Welch's method after interpolation.
        Since echo videos are short (typically 3-10s), frequency analysis is highly constrained.
        We interpolate to 4 Hz and calculate spectral power with safe guards.
        """
        n_beats = len(rr_sec)
        
        # Fallback for ultra-short recordings
        if n_beats < 4:
            self.log("Too few RR intervals for frequency-domain analysis. Returning default/estimated metrics.", level=30)
            return {"lf_ms2": 0.0, "hf_ms2": 0.0, "lf_hf": 1.0}
            
        # Cumulate time of beats to get timestamps
        timestamps = np.cumsum(rr_sec)
        
        # Interpolate to 4Hz
        fs_interp = 4.0
        t_start, t_end = timestamps[0], timestamps[-1]
        
        # Ensure we have a valid time span
        if t_end - t_start <= 0.5:
            return {"lf_ms2": 0.0, "hf_ms2": 0.0, "lf_hf": 1.0}
            
        t_new = np.arange(t_start, t_end, 1.0 / fs_interp)
        
        if len(t_new) < 4:
            return {"lf_ms2": 0.0, "hf_ms2": 0.0, "lf_hf": 1.0}
            
        try:
            # We interpolate the RR values (in seconds)
            f = interp1d(timestamps, rr_sec, kind='linear', fill_value="extrapolate")
            rr_interp = f(t_new)
            
            # Convert to ms and remove mean
            rr_interp_ms = (rr_interp - np.mean(rr_interp)) * 1000.0
            
            # Welch periodogram
            # Adjust nperseg based on available data length
            nperseg = min(len(rr_interp_ms), 64)
            freqs, psd = welch(rr_interp_ms, fs=fs_interp, nperseg=nperseg)
            
            # Identify frequency bands
            # LF: 0.04 - 0.15 Hz
            # HF: 0.15 - 0.40 Hz
            lf_mask = (freqs >= 0.04) & (freqs < 0.15)
            hf_mask = (freqs >= 0.15) & (freqs <= 0.40)
            
            # Integrate PSD to find power in ms^2
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
            self.log(f"Frequency domain calculation failed: {str(e)}. Using fallback.", level=30)
            return {"lf_ms2": 0.0, "hf_ms2": 0.0, "lf_hf": 1.0}

    def compute_poincare(self, rr_ms: np.ndarray) -> dict:
        """Computes Poincaré plot features (SD1, SD2, SD1/SD2)."""
        if len(rr_ms) < 3:
            return {"sd1": 0.0, "sd2": 0.0, "sd_ratio": 1.0}
            
        # Successive differences
        diff_rr = np.diff(rr_ms)
        
        # SD1: short-term HRV (perpendicular to identity line)
        sd1 = np.sqrt(0.5 * np.var(diff_rr, ddof=1))
        
        # SD2: long-term HRV (along identity line)
        # SD2^2 = 2*var(RR) - SD1^2
        var_rr = np.var(rr_ms, ddof=1)
        sd2_sq = 2.0 * var_rr - sd1**2
        sd2 = np.sqrt(max(0.0, sd2_sq))
        
        sd_ratio = sd1 / sd2 if sd2 > 0 else 1.0
        
        return {
            "sd1": float(sd1),
            "sd2": float(sd2),
            "sd_ratio": float(sd_ratio)
        }

    def classify_hrv_pattern(self, time_metrics: dict, poincare_metrics: dict, irregularity: float) -> str:
        """Heuristically classifies the HRV pattern based on clinical signatures."""
        sdnn = time_metrics["sdnn_ms"]
        rmssd = time_metrics["rmssd_ms"]
        sd_ratio = poincare_metrics["sd_ratio"]
        
        if irregularity > 0.15 and sdnn > 80.0 and sd_ratio > 0.4:
            return "afib_pattern"
        elif rmssd > 60.0 and irregularity > 0.08:
            return "pvc_pattern"
        else:
            return "normal"

    def execute(self, state: dict) -> dict:
        rr_sec = state["rr_intervals_sec"]
        if not rr_sec:
            raise ValueError("CardiacCycleAgent must execute before MechanicalHRVAgent.")
            
        rr_sec_np = np.array(rr_sec)
        rr_ms_np = rr_sec_np * 1000.0
        
        # Compute time, frequency, nonlinear domain features
        time_metrics = self.compute_time_domain(rr_ms_np)
        freq_metrics = self.compute_frequency_domain(rr_sec_np)
        poincare_metrics = self.compute_poincare(rr_ms_np)
        
        # Classify pattern
        pattern = self.classify_hrv_pattern(time_metrics, poincare_metrics, state["irregularity_index"])
        
        self.log(f"Mechanical HRV computed. SDNN={time_metrics['sdnn_ms']:.1f}ms, RMSSD={time_metrics['rmssd_ms']:.1f}ms, Pattern={pattern}")
        
        state["hrv_time"] = time_metrics
        state["hrv_freq"] = freq_metrics
        state["hrv_nonlinear"] = poincare_metrics
        state["hrv_pattern"] = pattern
        
        return state
