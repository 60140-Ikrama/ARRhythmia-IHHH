import os
import json
import numpy as np
from agents.base_agent import BaseAgent
from utils.state_manager import SharedMemory

class FeatureEngineeringAgent(BaseAgent):
    def __init__(self):
        super().__init__("FeatureEngineeringAgent", "Feature Fuser and Normalizer")
        self.scaler_path = "models/scaler_params.json"
        
        # Default scaling parameters (mean, std) for the 15 features
        self.default_scaler = {
            "ejection_fraction": {"mean": 55.0, "std": 10.0},
            "heart_rate_bpm": {"mean": 75.0, "std": 20.0},
            "irregularity_index": {"mean": 0.08, "std": 0.08},
            "sdnn_ms": {"mean": 60.0, "std": 30.0},
            "rmssd_ms": {"mean": 40.0, "std": 25.0},
            "pnn50": {"mean": 15.0, "std": 15.0},
            "lf_power": {"mean": 500.0, "std": 400.0},
            "hf_power": {"mean": 300.0, "std": 200.0},
            "lf_hf_ratio": {"mean": 1.8, "std": 1.0},
            "sd1": {"mean": 28.0, "std": 18.0},
            "sd2": {"mean": 75.0, "std": 35.0},
            "sd_ratio": {"mean": 0.45, "std": 0.2},
            "dyssynchrony_index_ms": {"mean": 50.0, "std": 35.0},
            "motion_irregularity": {"mean": 0.18, "std": 0.10},
            "hr_x_dyssynchrony": {"mean": 4000.0, "std": 3000.0}
        }

    def load_scaler(self) -> dict:
        """Loads scaler parameters from a JSON file if it exists."""
        if os.path.exists(self.scaler_path):
            try:
                with open(self.scaler_path, "r") as f:
                    scaler = json.load(f)
                self.log("Loaded pre-saved scaler parameters successfully.")
                return scaler
            except Exception as e:
                self.log(f"Failed to load scaler file: {str(e)}. Using defaults.", level=30)
        return self.default_scaler

    def execute(self, state: SharedMemory) -> SharedMemory:
        # Collect features from state
        ef = state.get("ejection_fraction", 0.0)
        
        # Check from MRVM
        mrvm = state.get("mrvm_features", {})
        hr = mrvm.get("heart_rate_bpm", state.get("heart_rate_bpm", 75.0))
        irreg = mrvm.get("irregularity_index", state.get("irregularity_index", 0.04))
        sdnn = mrvm.get("sdnn_ms", 0.0)
        rmssd = mrvm.get("rmssd_ms", 0.0)
        pnn50 = mrvm.get("pnn50", 0.0)
        lf = mrvm.get("lf_power", 0.0)
        hf = mrvm.get("hf_power", 0.0)
        lf_hf = mrvm.get("lf_hf_ratio", 1.0)
        sd1 = mrvm.get("sd1", 0.0)
        sd2 = mrvm.get("sd2", 0.0)
        sd_ratio = mrvm.get("sd_ratio", 1.0)
        
        # Check from Motion
        dys_index = state.get("dyssynchrony_index_ms", 0.0)
        mot_irreg = state.get("motion_irregularity", 0.0)
        
        # Interaction feature
        hr_x_dys = hr * dys_index
        
        feature_names = [
            "ejection_fraction",
            "heart_rate_bpm",
            "irregularity_index",
            "sdnn_ms",
            "rmssd_ms",
            "pnn50",
            "lf_power",
            "hf_power",
            "lf_hf_ratio",
            "sd1",
            "sd2",
            "sd_ratio",
            "dyssynchrony_index_ms",
            "motion_irregularity",
            "hr_x_dyssynchrony"
        ]
        
        raw_values = [
            ef, hr, irreg, sdnn, rmssd, pnn50, lf, hf, lf_hf, sd1, sd2, sd_ratio, dys_index, mot_irreg, hr_x_dys
        ]
        
        # Scale features
        scaler = self.load_scaler()
        norm_values = []
        
        for name, val in zip(feature_names, raw_values):
            mean = scaler.get(name, self.default_scaler[name])["mean"]
            std = scaler.get(name, self.default_scaler[name])["std"]
            std = std if std > 0 else 1.0
            norm_val = (val - mean) / std
            norm_values.append(norm_val)
            
        feature_vector = np.array(norm_values, dtype=np.float32)
        
        # Save back to state
        state.set("feature_vector", feature_vector.tolist())
        state.set("feature_names", feature_names)
        state.set("fused_feature_vector", feature_vector.tolist())
        
        # Log fused information
        self.log(f"Fused {len(feature_names)} features. Normalized Vector (first 5 elements): {norm_values[:5]}")
        return state

# Alias for compatibility with project manager
FeatureEngineeringAgent = FeatureEngineeringAgent
