import numpy as np
from agents.base_agent import BaseAgent
from utils.state_manager import SharedMemory

class UncertaintyAgent(BaseAgent):
    def __init__(self):
        super().__init__("UncertaintyAgent", "Quality and Confidence Auditor")

    def execute(self, state: SharedMemory) -> SharedMemory:
        self.log("Evaluating diagnostic uncertainty and calibrating confidence...")
        
        # Read from config if available
        configs = state.get("configs", {})
        u_config = configs.get("uncertainty", {
            "confidence_threshold": 0.70,
            "ood_threshold": 3.0
        })
        
        # Get raw predictions
        preds = state.get("predictions", {})
        rhythm = preds.get("rhythm", "unknown")
        raw_confidence = preds.get("confidence", 0.0)
        
        # 1. Platt calibration simulation
        # For this prototype, we calibrate by combining prediction confidence with data quality scores
        quality = state.get("quality_metrics", {})
        passed_quality = quality.get("passed", True)
        noise_snr = quality.get("noise_snr", 15.0)
        
        # Deprecate confidence if SNR is extremely low (indicative of poor image quality)
        snr_penalty = 1.0 if noise_snr >= 8.0 else (noise_snr / 8.0)
        calibrated_confidence = raw_confidence * snr_penalty
        
        # 2. Out-of-distribution detection (OOD)
        # We calculate Mahalanobis-like distance using normalized feature vectors
        fused_features = state.get("fused_feature_vector", [])
        if len(fused_features) > 0:
            # Distance from center of standardized feature space (mean = 0, std = 1)
            dist_sq = np.sum(np.square(fused_features))
            mean_dist = np.sqrt(dist_sq) / len(fused_features)
            ood_flag = mean_dist > u_config["ood_threshold"]
        else:
            mean_dist = 0.0
            ood_flag = False
            
        # 3. Clinical review trigger rules
        requires_review = False
        if calibrated_confidence < u_config["confidence_threshold"]:
            self.log(f"Low confidence ({calibrated_confidence*100:.1f}%) triggers manual review.", level=30)
            requires_review = True
            
        if not passed_quality:
            self.log("Failed input video quality triggers manual review.", level=30)
            requires_review = True
            
        if ood_flag:
            self.log(f"Out-of-Distribution feature structure (distance={mean_dist:.2f}) triggers manual review.", level=30)
            requires_review = True
            
        self.log(f"Calibrated Confidence: {calibrated_confidence*100:.1f}%, OOD Flag: {ood_flag}, Requires Review: {requires_review}")
        
        uncertainty_metrics = {
            "calibrated_confidence": float(calibrated_confidence),
            "ood_flag": bool(ood_flag),
            "requires_review": bool(requires_review)
        }
        
        state.set("uncertainty_metrics", uncertainty_metrics)
        # Update legacy keys for backward compatibility
        state.set("requires_review", bool(requires_review))
        state.set("confidence", float(calibrated_confidence))
        
        return state
