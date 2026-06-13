import os
import json
import numpy as np
from agents.base_agent import BaseAgent

class FeatureEngineeringAgent(BaseAgent):
    def __init__(self):
        super().__init__("FeatureEngineeringAgent", "Feature Fuser and Normalizer")
        self.scaler_path = "models/scaler_params.json"
        
        # Default scaling parameters (mean, std) for the 15 features
        # derived from clinical reference ranges and synthetic distributions
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
        """Loads scaler parameters from a JSON file if it exists, otherwise returns default scaling parameters."""
        if os.path.exists(self.scaler_path):
            try:
                with open(self.scaler_path, "r") as f:
                    scaler = json.load(f)
                self.log("Loaded pre-saved scaler parameters successfully.")
                return scaler
            except Exception as e:
                self.log(f"Failed to load scaler file: {str(e)}. Using defaults.", level=30)
        return self.default_scaler

    def save_scaler(self, scaler_params: dict):
        """Saves scaler parameters to a JSON file."""
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
        try:
            with open(self.scaler_path, "w") as f:
                json.dump(scaler_params, f, indent=4)
            self.log(f"Saved scaler parameters to {self.scaler_path}")
        except Exception as e:
            self.log(f"Failed to save scaler file: {str(e)}", level=30)

    def execute(self, state: dict) -> dict:
        # 1. Collect features from upstream agents
        ef = state.get("ejection_fraction", 0.0)
        hr = state.get("heart_rate_bpm", 0.0)
        irreg = state.get("irregularity_index", 0.0)
        
        # HRV
        hrv_time = state.get("hrv_time", {})
        sdnn = hrv_time.get("sdnn_ms", 0.0)
        rmssd = hrv_time.get("rmssd_ms", 0.0)
        pnn50 = hrv_time.get("pnn50", 0.0)
        
        hrv_freq = state.get("hrv_freq", {})
        lf = hrv_freq.get("lf_ms2", 0.0)
        hf = hrv_freq.get("hf_ms2", 0.0)
        lf_hf = hrv_freq.get("lf_hf", 1.0)
        
        hrv_nonlinear = state.get("hrv_nonlinear", {})
        sd1 = hrv_nonlinear.get("sd1", 0.0)
        sd2 = hrv_nonlinear.get("sd2", 0.0)
        sd_ratio = hrv_nonlinear.get("sd_ratio", 1.0)
        
        # Motion
        dys_index = state.get("dyssynchrony_index_ms", 0.0)
        mot_irreg = state.get("motion_irregularity", 0.0)
        
        # 2. Build feature vector (14 base features + 1 interaction feature)
        # Interaction feature: hr * dyssynchrony
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
        
        # 3. Normalize features using loaded/default scaler
        scaler = self.load_scaler()
        norm_values = []
        
        for name, val in zip(feature_names, raw_values):
            mean = scaler[name]["mean"]
            std = scaler[name]["std"]
            # Prevent division by zero
            std = std if std > 0 else 1.0
            norm_val = (val - mean) / std
            norm_values.append(norm_val)
            
        feature_vector = np.array(norm_values, dtype=np.float32)
        
        self.log(f"Fused 15 features. Raw Vector: {raw_values}")
        self.log(f"Normalized Vector (first 5 elements): {norm_values[:5]}")
        
        state["feature_vector"] = feature_vector.tolist()
        state["feature_names"] = feature_names
        
        return state
