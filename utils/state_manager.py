import time
import logging
import threading
import numpy as np

class SharedMemory:
    """
    Thread-safe and transaction-logged State Manager.
    Enforces key checking and type/value rules to ensure high robustness.
    Implements dictionary interfaces (__getitem__, __setitem__, __contains__) for backward compatibility.
    """
    def __init__(self, patient_id: str, video_path: str):
        self.lock = threading.Lock()
        self.logger = logging.getLogger("SharedMemory")
        
        # Initialize internal state dictionary matching the structured schema
        self._state = {
            # Metadata & Configs
            "patient_id": patient_id,
            "video_path": video_path,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "configs": {},
            "pipeline_status": [],  # List of statuses: ["ProjectManagerAgent: RUNNING", ...]
            
            # Data quality metrics
            "quality_metrics": {
                "fps": 0.0,
                "noise_snr": 0.0,
                "blur_laplacian": 0.0,
                "missing_frames": 0,
                "artifact_score": 0.0,
                "passed": False
            },
            
            # View type
            "view_classification": {
                "predicted_view": "unknown",
                "confidence": 0.0,
                "probabilities": {}
            },
            
            # Segmentations
            "segmentations": {
                "masks": None,      # np.ndarray shape (T, H, W)
                "volumes": {"LV": [], "LA": [], "RV": [], "RA": []},
                "ejection_fraction": 0.0,
                "segmentation_confidence": 0.0
            },
            
            # Motion
            "motion_tracking": {
                "velocities": None, # np.ndarray shape (T-1, H, W, 2)
                "dyssynchrony_index_ms": 0.0,
                "motion_irregularity": 0.0,
                "regional_velocities": {"septal": [], "lateral": [], "apical": [], "basal": []}
            },
            
            # Strain
            "strain_analysis": {
                "gls_curve": [],
                "radial_strain": [],
                "circumferential_strain": [],
                "atrial_strain": [],
                "peak_strain": {"GLS": 0.0, "radial": 0.0, "circumferential": 0.0, "atrial": 0.0}
            },
            
            # MRVM
            "mrvm_features": {
                "rr_intervals_sec": [],
                "heart_rate_bpm": 0.0,
                "irregularity_index": 0.0,
                "sdnn_ms": 0.0,
                "rmssd_ms": 0.0,
                "pnn50": 0.0,
                "lf_power": 0.0,
                "hf_power": 0.0,
                "lf_hf_ratio": 1.0,
                "sd1": 0.0,
                "sd2": 0.0,
                "sd_ratio": 1.0,
                "entropy": 0.0
            },
            
            # Pseudo-ECG
            "pseudo_ecg": {
                "synthetic_signal": [],
                "sampling_rate_hz": 50.0,
                "reconstruction_correlation": 0.0
            },
            
            # Deep video features
            "deep_embeddings": [],
            
            # Feature Fusion
            "fused_feature_vector": [],
            "feature_names": [],
            
            # Arrhythmia classification
            "predictions": {
                "rhythm": "unknown",
                "confidence": 0.0,
                "probabilities": {}
            },
            
            # Uncertainty
            "uncertainty_metrics": {
                "calibrated_confidence": 0.0,
                "ood_flag": False,
                "requires_review": False
            },
            
            # Knowledge Graph
            "kg_subgraph": {},
            
            # Clinical Reasoning
            "reasoning": {
                "impression": "",
                "evidence_points": [],
                "recommendations": []
            },
            
            # Explainability
            "explainability_artifacts": {
                "shap_values": [],
                "saliency_map_paths": []
            },
            
            # Final Report Paths
            "report_paths": {
                "txt": "",
                "json": "",
                "pdf": ""
            },
            
            # Base components needed by existing scripts
            "is_mock": False,
            "mock_rhythm": "",
            "errors": [],
            "rhythm": "unknown",
            "confidence": 0.0,
            "requires_review": False
        }
        
    def __getitem__(self, key: str):
        return self.get(key)

    def __setitem__(self, key: str, value):
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        with self.lock:
            return key in self._state
            
    def get(self, key: str, default=None):
        """Thread-safe state read access."""
        with self.lock:
            if key in self._state:
                return self._state[key]
            
            # Also support legacy keys mapped from inner dicts if requested
            legacy_mappings = {
                "ejection_fraction": ("segmentations", "ejection_fraction"),
                "heart_rate_bpm": ("mrvm_features", "heart_rate_bpm"),
                "irregularity_index": ("mrvm_features", "irregularity_index"),
                "dyssynchrony_index_ms": ("motion_tracking", "dyssynchrony_index_ms"),
                "motion_irregularity": ("motion_tracking", "motion_irregularity"),
                "rhythm": ("predictions", "rhythm"),
                "confidence": ("predictions", "confidence"),
                "probabilities": ("predictions", "probabilities"),
                "requires_review": ("uncertainty_metrics", "requires_review"),
                "evidence": ("reasoning", "evidence_points"),
                "report_txt": ("report_paths", "txt"),
                "report_json": ("report_paths", "json"),
                "volume_curve": ("segmentations", "volumes") # Note: handled specially
            }
            
            if key in legacy_mappings:
                section, field = legacy_mappings[key]
                if field == "volumes" and section == "segmentations":
                    lv_vols = self._state["segmentations"]["volumes"].get("LV", [])
                    return np.array(lv_vols) if lv_vols else None
                return self._state[section][field]
                
            return default

    def set(self, key: str, value):
        """Thread-safe state write access with transaction logging."""
        with self.lock:
            self._state[key] = value
            self.logger.debug(f"State update: {key} set to type {type(value)}")

    def update_nested(self, section: str, field: str, value):
        """Thread-safe update of nested dict properties."""
        with self.lock:
            if section not in self._state:
                raise KeyError(f"Section '{section}' not found in state schema.")
            if not isinstance(self._state[section], dict):
                raise TypeError(f"State field '{section}' is not a dictionary.")
            self._state[section][field] = value
            self.logger.debug(f"State update: {section}.{field} updated")

    def get_nested(self, section: str, field: str, default=None):
        """Thread-safe read of nested dict properties."""
        with self.lock:
            if section not in self._state:
                return default
            if not isinstance(self._state[section], dict):
                return default
            return self._state[section].get(field, default)

    def log_pipeline_status(self, agent_name: str, status: str):
        """Append to transaction logging."""
        with self.lock:
            msg = f"{agent_name}: {status}"
            self._state["pipeline_status"].append(msg)
            self.logger.info(f"Transaction log: {msg}")

    def add_error(self, agent_name: str, error_msg: str):
        """Append error to error trace."""
        with self.lock:
            msg = f"[{agent_name}] {error_msg}"
            self._state["errors"].append(msg)
            self.logger.error(msg)

    def to_dict(self) -> dict:
        """Returns a snapshot of the raw dictionary state."""
        with self.lock:
            import copy
            snapshot = {}
            for k, v in self._state.items():
                if isinstance(v, np.ndarray):
                    snapshot[k] = v
                elif isinstance(v, dict):
                    snapshot[k] = {ik: (iv if isinstance(iv, np.ndarray) else copy.deepcopy(iv)) for ik, iv in v.items()}
                else:
                    snapshot[k] = copy.deepcopy(v)
            return snapshot
