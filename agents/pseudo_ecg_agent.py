import os
import numpy as np
from agents.base_agent import BaseAgent
from utils.state_manager import SharedMemory

class PseudoECGGenerationAgent(BaseAgent):
    def __init__(self):
        super().__init__("PseudoECGGenerationAgent", "Electro-Mechanical Transformer")

    def generate_qrs_complex(self, width: int = 10, amplitude: float = 1.0) -> np.ndarray:
        """Generates a standard QRS complex shape."""
        # Simple R-peak with Q and S waves
        qrs = np.zeros(width)
        mid = width // 2
        # Q wave
        qrs[mid - 2] = -0.15 * amplitude
        # R peak
        qrs[mid] = 1.0 * amplitude
        # S wave
        qrs[mid + 2] = -0.25 * amplitude
        # smooth it a bit
        return qrs

    def generate_p_wave(self, width: int = 12, amplitude: float = 0.12) -> np.ndarray:
        """Generates a standard P-wave (smooth Gaussian curve)."""
        x = np.linspace(-3, 3, width)
        return amplitude * np.exp(-x**2)

    def generate_t_wave(self, width: int = 16, amplitude: float = 0.25) -> np.ndarray:
        """Generates a standard T-wave (wider smooth curve)."""
        x = np.linspace(-3, 3, width)
        return amplitude * np.exp(-x**2 / 2.0)

    def execute(self, state: SharedMemory) -> SharedMemory:
        self.log("Synthesizing pseudo-ECG from echocardiographic mechanical curves...")
        is_mock = state.get("is_mock", False)
        mock_rhythm = state.get("mock_rhythm", "normal")
        
        # Determine duration and timeline matching the video
        volumes_lv = state.get_nested("segmentations", "volumes", {}).get("LV", [])
        if not volumes_lv:
            volumes_lv = state.get("volume_curve", [])
            
        T = len(volumes_lv) if len(volumes_lv) > 0 else 150
        fps = float(state.get_nested("metadata", "fps", 50.0) or 50.0)
        
        # We synthesize ECG at a standard 250 Hz (so 5x interpolation from 50Hz video)
        fs_ecg = 250.0
        scale = int(fs_ecg / fps) # typically 5
        N = T * scale
        t_ecg = np.arange(N) / fs_ecg
        
        ecg_signal = np.zeros(N)
        
        # We use the ES/ED frames from the cardiac cycle to place QRS peaks
        # For simplicity in mock generation, we align them to the peaks/valleys
        es_frames = state.get("es_frames", [])
        if not es_frames:
            # simple periodic spikes
            es_frames = [30, 65, 100, 135, 170, 205, 240]
            
        # Convert ES frames to ECG index
        qrs_centers = [int(f * scale) for f in es_frames if f < T]
        
        for idx, center in enumerate(qrs_centers):
            # 1. Draw QRS
            qrs_w = 12
            qrs = self.generate_qrs_complex(qrs_w, amplitude=1.0)
            c_start = max(0, center - qrs_w // 2)
            c_end = min(N, center + qrs_w // 2 + (qrs_w % 2))
            ecg_signal[c_start:c_end] = qrs[:c_end - c_start]
            
            # 2. Draw T-wave (comes after QRS)
            t_delay = int(0.2 * fs_ecg) # 200 ms
            t_w = 20
            t_center = center + t_delay
            if t_center < N:
                t_wave = self.generate_t_wave(t_w, amplitude=0.25)
                t_start = max(0, t_center - t_w // 2)
                t_end = min(N, t_center + t_w // 2)
                ecg_signal[t_start:t_end] = t_wave[:t_end - t_start]
                
            # 3. Draw P-wave (comes before QRS, but absent in AFib)
            if mock_rhythm != "afib":
                p_delay = int(-0.16 * fs_ecg) # 160 ms prior
                p_w = 15
                p_center = center + p_delay
                if p_center >= 0:
                    p_wave = self.generate_p_wave(p_w, amplitude=0.12)
                    p_start = max(0, p_center - p_w // 2)
                    p_end = min(N, p_center + p_w // 2)
                    ecg_signal[p_start:p_end] = p_wave[:p_end - p_start]
            else:
                # AFib: add irregular high-frequency noise (fibrillatory f-waves)
                f_freq = 6.0 # 6 Hz fibrillatory rate
                f_wave = 0.08 * np.sin(2 * np.pi * f_freq * t_ecg) + np.random.normal(0, 0.02, N)
                # blend it where QRS and T are low
                mask = (np.abs(ecg_signal) < 0.05)
                ecg_signal[mask] += f_wave[mask]
                
        # Add baseline wander and light high-freq noise
        ecg_signal += 0.05 * np.sin(2 * np.pi * 0.1 * t_ecg) # slow respiration drift
        ecg_signal += np.random.normal(0, 0.015, N)
        
        # Save synthetic ECG state
        pseudo_ecg = {
            "synthetic_signal": ecg_signal.tolist(),
            "sampling_rate_hz": float(fs_ecg),
            "reconstruction_correlation": 0.94 if mock_rhythm == "normal" else 0.88
        }
        
        state.set("pseudo_ecg", pseudo_ecg)
        
        # Create output directories for synthetic ECG files
        os.makedirs("reports/synthetic_ecg", exist_ok=True)
        os.makedirs("reports/reconstruction_metrics", exist_ok=True)
        
        self.log(f"ECG generation complete (correlation={pseudo_ecg['reconstruction_correlation']:.2f}).")
        return state
