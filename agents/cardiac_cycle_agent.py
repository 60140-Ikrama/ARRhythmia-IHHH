import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter
from agents.base_agent import BaseAgent


class CardiacCycleAgent(BaseAgent):
    def __init__(self):
        super().__init__("CardiacCycleAgent", "Robust ED/ES Cycle Analysis (Clinical Grade)")

    # =========================
    # 1. SIGNAL PREPROCESSING
    # =========================
    def preprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        """Smooth + normalize LV volume curve for robust peak detection."""
        if len(signal) < 7:
            return signal

        # Savitzky-Golay smoothing (critical improvement)
        window = min(11, len(signal) // 2 * 2 + 1)
        if window >= 5:
            signal = savgol_filter(signal, window_length=window, polyorder=2)

        # Normalize (removes amplitude scaling issues)
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

        return signal

    # =========================
    # 2. BANDPASS FILTER
    # =========================
    def bandpass_filter(self, signal, low=0.5, high=5.0, fs=50.0):
        nyq = 0.5 * fs
        low_n = low / nyq
        high_n = high / nyq

        order = 2 if len(signal) > 20 else 1
        b, a = butter(order, [low_n, high_n], btype='band')

        try:
            return filtfilt(b, a, signal)
        except:
            return signal

    # =========================
    # 3. ROBUST PEAK DETECTION
    # =========================
    def detect_peaks(self, v, fs):
        """Detect ED and ES candidates robustly."""

        prom = np.std(v) * 0.4
        min_dist = max(2, int(0.25 * fs))  # physiological constraint

        ed, _ = find_peaks(v, distance=min_dist, prominence=prom)
        es, _ = find_peaks(-v, distance=min_dist, prominence=prom)

        # fallback if too few cycles
        if len(ed) < 2 or len(es) < 2:
            ed, _ = find_peaks(v, distance=min_dist, prominence=prom * 0.6)
            es, _ = find_peaks(-v, distance=min_dist, prominence=prom * 0.6)

        return ed, es

    # =========================
    # 4. CYCLE PAIRING (KEY FIX)
    # =========================
    def build_cycles(self, ed, es):
        """
        Enforces physiological structure:
        ED → ES → ED cycles
        """
        cycles = []

        for i in range(len(ed) - 1):
            ed_start = ed[i]
            ed_end = ed[i + 1]

            # ES must lie between EDs
            valid_es = es[(es > ed_start) & (es < ed_end)]

            if len(valid_es) == 0:
                continue

            es_peak = valid_es[np.argmin(valid_es - ed_start)]  # closest ES

            cycles.append((ed_start, es_peak, ed_end))

        return cycles

    # =========================
    # 5. CYCLE QUALITY SCORE
    # =========================
    def cycle_quality(self, v, ed, es):
        amp = abs(v[ed] - v[es])
        duration = es - ed

        if duration <= 0:
            return 0

        return amp / duration  # simple but effective reliability metric

    # =========================
    # 6. MAIN EXECUTION
    # =========================
    def execute(self, state: dict):

        volume_curve = np.array(state["volume_curve"])
        if len(volume_curve) == 0:
            raise ValueError("Missing volume curve")

        fs = float(state["metadata"].get("fps", 50.0))

        # Step 1: preprocess
        v = self.preprocess_signal(volume_curve)

        # Step 2: bandpass filter
        v = self.bandpass_filter(v, fs=fs)

        # Step 3: detect peaks
        ed, es = self.detect_peaks(v, fs)

        if len(ed) < 2 or len(es) < 2:
            raise ValueError("Insufficient cardiac structure detected")

        # Step 4: build physiological cycles
        cycles = self.build_cycles(ed, es)

        if len(cycles) < 2:
            raise ValueError("Failed to construct valid cardiac cycles")

        # =========================
        # 7. FEATURE EXTRACTION
        # =========================

        rr_intervals = []
        ef_values = []
        quality_scores = []

        for ed1, es1, ed2 in cycles:

            rr_intervals.append((ed2 - ed1) / fs)

            edv = volume_curve[ed1]
            esv = volume_curve[es1]

            if edv > 0:
                ef = ((edv - esv) / edv) * 100
                ef_values.append(np.clip(ef, 0, 100))

            quality_scores.append(self.cycle_quality(v, ed1, es1))

        rr_intervals = np.array(rr_intervals)

        # =========================
        # 8. CLINICAL METRICS
        # =========================

        mean_rr = np.mean(rr_intervals)
        hr = 60 / mean_rr if mean_rr > 0 else 0

        irregularity = np.std(rr_intervals) / (mean_rr + 1e-8)

        ef_mean = np.mean(ef_values) if len(ef_values) else 0

        cycle_confidence = np.mean(quality_scores)

        # =========================
        # 9. LOGGING
        # =========================

        self.log(
            f"Cycle Analysis Complete | HR={hr:.1f} BPM | EF={ef_mean:.1f}% | "
            f"Irregularity={irregularity:.3f} | Confidence={cycle_confidence:.3f}"
        )

        # =========================
        # 10. SAVE STATE
        # =========================

        state.update({
            "ed_frames": ed.tolist(),
            "es_frames": es.tolist(),
            "rr_intervals_sec": rr_intervals.tolist(),
            "heart_rate_bpm": float(hr),
            "ejection_fraction": float(ef_mean),
            "irregularity_index": float(irregularity),
            "cycle_confidence": float(cycle_confidence),
            "num_cycles": len(cycles)
        })

        return state
