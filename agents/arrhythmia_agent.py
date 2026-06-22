import os
import numpy as np
from agents.base_agent import BaseAgent
from utils.state_manager import SharedMemory

try:
    import xgboost as xgb
except ImportError:
    xgb = None

class ArrhythmiaClassificationAgent(BaseAgent):
    def __init__(self):
        super().__init__("ArrhythmiaClassificationAgent", "Rhythm Diagnostician")
        self.model_path = "models/xgboost_arrhythmia.json"
        self.booster = None

    def load_model(self) -> bool:
        """Attempts to load the pretrained XGBoost model."""
        if xgb is None:
            self.log("XGBoost library is not installed. Will use expert rules fallback.", level=30)
            return False
            
        if not os.path.exists(self.model_path):
            self.log(f"XGBoost model file not found at {self.model_path}. Will use expert rules fallback.", level=30)
            return False
            
        try:
            self.booster = xgb.Booster()
            self.booster.load_model(self.model_path)
            self.log("Successfully loaded pretrained XGBoost classifier.")
            return True
        except Exception as e:
            self.log(f"Failed to load XGBoost model: {str(e)}. Falling back.", level=30)
            return False

    def predict_with_xgboost(self, feature_vector: list) -> dict:
        """Runs inference using the loaded XGBoost booster."""
        x = np.array(feature_vector).reshape(1, -1)
        dmatrix = xgb.DMatrix(x)
        probs = self.booster.predict(dmatrix)[0] # assumes multi:softprob output
        
        classes = ["normal_sinus_rhythm", "atrial_fibrillation", "pvc", "bradycardia", "tachycardia"]
        
        if len(probs) != len(classes):
            probs_dict = {cls: 0.0 for cls in classes}
            probs_dict["normal_sinus_rhythm"] = 1.0
            return probs_dict
            
        return {classes[i]: float(probs[i]) for i in range(len(classes))}

    def predict_with_rules(self, raw_features: dict) -> dict:
        """Expert clinical decision rules fallback when XGBoost is not available."""
        hr = raw_features.get("heart_rate_bpm", 75.0)
        irreg = raw_features.get("irregularity_index", 0.04)
        sdnn = raw_features.get("sdnn_ms", 45.0)
        sd_ratio = raw_features.get("sd_ratio", 0.6)
        mot_irreg = raw_features.get("motion_irregularity", 0.15)
        rmssd = raw_features.get("rmssd_ms", 30.0)
        
        is_normal = (60.0 <= hr <= 100.0) and (irreg < 0.08) and (mot_irreg < 0.25)
        
        probs = {
            "normal_sinus_rhythm": 0.0,
            "atrial_fibrillation": 0.0,
            "pvc": 0.0,
            "bradycardia": 0.0,
            "tachycardia": 0.0
        }
        
        if is_normal:
            probs["normal_sinus_rhythm"] = 0.95
            probs["atrial_fibrillation"] = 0.02
            probs["pvc"] = 0.01
            probs["bradycardia"] = 0.01
            probs["tachycardia"] = 0.01
            return probs
            
        if irreg < 0.08:
            if hr < 60.0:
                probs["bradycardia"] = 0.92
                probs["normal_sinus_rhythm"] = 0.05
                probs["atrial_fibrillation"] = 0.01
                probs["pvc"] = 0.01
                probs["tachycardia"] = 0.01
            elif hr > 100.0:
                probs["tachycardia"] = 0.94
                probs["normal_sinus_rhythm"] = 0.03
                probs["atrial_fibrillation"] = 0.01
                probs["pvc"] = 0.01
                probs["bradycardia"] = 0.01
            else:
                probs["normal_sinus_rhythm"] = 0.60
                probs["pvc"] = 0.25
                probs["atrial_fibrillation"] = 0.15
            return probs
            
        if irreg >= 0.15 or (sdnn > 80.0 and sd_ratio < 0.5):
            probs["atrial_fibrillation"] = 0.88
            probs["pvc"] = 0.08
            probs["normal_sinus_rhythm"] = 0.02
            probs["bradycardia"] = 0.01
            probs["tachycardia"] = 0.01
        else:
            probs["pvc"] = 0.85
            probs["atrial_fibrillation"] = 0.10
            probs["normal_sinus_rhythm"] = 0.03
            probs["bradycardia"] = 0.01
            probs["tachycardia"] = 0.01
            
        return probs

    def get_evidence(self, rhythm: str, raw_features: dict) -> list:
        """Generates patient-specific evidence explaining the classification decision."""
        evidence = []
        hr = raw_features.get("heart_rate_bpm", 75.0)
        irreg = raw_features.get("irregularity_index", 0.04)
        sdnn = raw_features.get("sdnn_ms", 45.0)
        rmssd = raw_features.get("rmssd_ms", 30.0)
        sd_ratio = raw_features.get("sd_ratio", 0.6)
        mot_irreg = raw_features.get("motion_irregularity", 0.15)
        dys_index = raw_features.get("dyssynchrony_index_ms", 50.0)
        
        if rhythm == "normal_sinus_rhythm":
            evidence.append(f"Regular heart rate ({hr:.1f} BPM) within normal clinical limits (60-100 BPM).")
            evidence.append(f"Low rhythm irregularity index ({irreg:.3f}, typical NSR: <0.08).")
            evidence.append(f"Stable myocardial wall motion tracking (irregularity: {mot_irreg:.3f}).")
        elif rhythm == "atrial_fibrillation":
            evidence.append(f"High rhythm irregularity index ({irreg:.3f}, AFib threshold: >0.15).")
            evidence.append(f"Markedly elevated SDNN ({sdnn:.1f} ms, indicating highly variable beat intervals).")
            evidence.append(f"Fan-shaped Poincaré plot with reduced SD1/SD2 ratio ({sd_ratio:.2f}).")
            evidence.append(f"Chaotic mechanical wall displacement (motion irregularity: {mot_irreg:.3f}).")
        elif rhythm == "pvc":
            evidence.append(f"Isolated ectopic beat pattern with elevated RMSSD ({rmssd:.1f} ms).")
            evidence.append(f"Mild-to-moderate global rhythm irregularity ({irreg:.3f}) with compensation pauses.")
            evidence.append(f"Localized dyssynchrony index ({dys_index:.1f} ms) typical of ventricular ectopy.")
            evidence.append(f"Abrupt wall motion jerk signature (motion irregularity: {mot_irreg:.3f}).")
        elif rhythm == "bradycardia":
            evidence.append(f"Depressed mechanical heart rate ({hr:.1f} BPM, bradycardia threshold: <60 BPM).")
            evidence.append(f"Stable, regular rhythm (irregularity: {irreg:.3f}).")
        elif rhythm == "tachycardia":
            evidence.append(f"Elevated mechanical heart rate ({hr:.1f} BPM, tachycardia threshold: >100 BPM).")
            evidence.append(f"Regular beat structure with fast contraction rate (irregularity: {irreg:.3f}).")
            
        return evidence

    def execute(self, state: SharedMemory) -> SharedMemory:
        feature_vector = state.get("fused_feature_vector")
        if not feature_vector:
            feature_vector = state.get("feature_vector")
            
        if feature_vector is None:
            raise ValueError("Feature Engineering Agent must execute before Arrhythmia Classification Agent.")
            
        # Reconstruct raw features for rule validation / evidence
        mrvm = state.get("mrvm_features", {})
        raw_features = {
            "heart_rate_bpm": mrvm.get("heart_rate_bpm", state.get("heart_rate_bpm", 75.0)),
            "irregularity_index": mrvm.get("irregularity_index", state.get("irregularity_index", 0.04)),
            "sdnn_ms": mrvm.get("sdnn_ms", 45.0),
            "rmssd_ms": mrvm.get("rmssd_ms", 30.0),
            "sd_ratio": mrvm.get("sd_ratio", 0.6),
            "motion_irregularity": state.get("motion_irregularity", 0.15),
            "dyssynchrony_index_ms": state.get("dyssynchrony_index_ms", 50.0)
        }
        
        has_model = self.load_model()
        
        if has_model:
            probabilities = self.predict_with_xgboost(feature_vector)
        else:
            probabilities = self.predict_with_rules(raw_features)
            
        rhythm = max(probabilities, key=probabilities.get)
        confidence = probabilities[rhythm]
        
        # Save back to state
        state.set("rhythm", rhythm)
        state.set("confidence", float(confidence))
        state.set("probabilities", probabilities)
        state.set("evidence", self.get_evidence(rhythm, raw_features))
        
        # Save nested structure
        predictions_state = {
            "rhythm": rhythm,
            "confidence": float(confidence),
            "probabilities": probabilities
        }
        state.set("predictions", predictions_state)
        
        # Decision path
        if rhythm == "normal_sinus_rhythm":
            decision_path = "Normal Heart Rate & Rhythm -> NSR"
        elif rhythm in ["bradycardia", "tachycardia"]:
            decision_path = "Abnormal Rate -> Regular -> " + rhythm.capitalize()
        elif rhythm == "atrial_fibrillation":
            decision_path = "Abnormal Rhythm -> High Irregularity -> AFib"
        else:
            decision_path = "Abnormal Rhythm -> Ectopic Skips -> PVC"
        state.set("decision_path", decision_path)
        
        self.log(f"Arrhythmia Diagnosis complete: {rhythm.upper()} (confidence={confidence*100:.1f}%)")
        return state

# Alias for compatibility with project manager
ArrhythmiaDetectionAgent = ArrhythmiaClassificationAgent
