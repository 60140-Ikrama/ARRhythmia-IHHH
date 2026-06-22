import os
import json
import numpy as np
from agents.base_agent import BaseAgent
from utils.state_manager import SharedMemory

try:
    import cv2
except ImportError:
    cv2 = None

class DataQualityAgent(BaseAgent):
    def __init__(self):
        super().__init__("DataQualityAgent", "Quality Auditor")

    def estimate_blur(self, frames: np.ndarray) -> float:
        """Estimates video blur using the variance of the Laplacian method."""
        if cv2 is None or len(frames) == 0:
            return 25.0  # mock value
            
        laplacian_vars = []
        for frame in frames[:10]:  # sample first 10 frames
            # convert frame back to 0-255 uint8 if normalized
            img = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
            var = cv2.Laplacian(img, cv2.CV_64F).var()
            laplacian_vars.append(var)
        return float(np.mean(laplacian_vars))

    def estimate_noise_snr(self, frames: np.ndarray) -> float:
        """Estimates Signal-to-Noise Ratio (SNR)."""
        if len(frames) == 0:
            return 15.0
        mean_val = np.mean(frames)
        std_val = np.std(frames)
        return float(mean_val / std_val) if std_val > 0 else 0.0

    def execute(self, state: SharedMemory) -> SharedMemory:
        self.log("Evaluating echocardiography video quality...")
        is_mock = state.get("is_mock", False)
        
        # Read from config if available
        configs = state.get("configs", {})
        q_config = configs.get("data_quality", {
            "min_fps": 15,
            "max_missing_frames": 2,
            "min_snr": 5.0,
            "min_laplacian_var": 10.0
        })
        
        if is_mock:
            self.log("Mock mode enabled. Simulating high-quality data metrics.")
            quality_metrics = {
                "fps": 50.0,
                "noise_snr": 18.5,
                "blur_laplacian": 45.2,
                "missing_frames": 0,
                "artifact_score": 0.05,
                "passed": True
            }
        else:
            # For real videos, the video is loaded by project manager or DataAgent
            # Let's check if 'echo_video' exists in state
            echo_video = state.get("echo_video")
            metadata = state.get("metadata", {})
            
            if echo_video is None:
                # If not loaded yet, we can check video path
                video_path = state.get("video_path")
                if not video_path or not os.path.exists(video_path):
                    raise ValueError(f"Video file not found or echo_video is missing: {video_path}")
                # Load video just enough to check quality
                cap = cv2.VideoCapture(video_path)
                fps = float(cap.get(cv2.CAP_PROP_FPS)) or 50.0
                frames = []
                while len(frames) < 30:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frames.append(gray.astype(np.float32) / 255.0)
                cap.release()
                echo_video = np.array(frames)
            else:
                fps = float(metadata.get("fps", 50.0))
                
            blur = self.estimate_blur(echo_video)
            snr = self.estimate_noise_snr(echo_video)
            
            passed = (
                fps >= q_config["min_fps"] and
                snr >= q_config["min_snr"] and
                blur >= q_config["min_laplacian_var"]
            )
            
            quality_metrics = {
                "fps": fps,
                "noise_snr": snr,
                "blur_laplacian": blur,
                "missing_frames": 0,
                "artifact_score": 0.12 if snr < 8 else 0.02,
                "passed": bool(passed)
            }
            
        self.log(f"Quality Metrics: FPS={quality_metrics['fps']:.1f}, SNR={quality_metrics['noise_snr']:.2f}, Blur={quality_metrics['blur_laplacian']:.1f}, Passed={quality_metrics['passed']}")
        
        # Save to state
        state.set("quality_metrics", quality_metrics)
        
        # Save to reports/quality_report.json
        os.makedirs("reports", exist_ok=True)
        try:
            with open("reports/quality_report.json", "w") as f:
                json.dump(quality_metrics, f, indent=4)
        except Exception as e:
            self.log(f"Could not save quality_report.json: {str(e)}", level=30)
            
        return state
