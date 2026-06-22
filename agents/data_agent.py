import os
import numpy as np
from agents.base_agent import BaseAgent
from utils.state_manager import SharedMemory

try:
    import cv2
except ImportError:
    cv2 = None

class DataAgent(BaseAgent):
    def __init__(self):
        super().__init__("DataAgent", "Loader and Preprocessor")

    def execute(self, state: SharedMemory) -> SharedMemory:
        video_path = state.get("video_path")
        is_mock = state.get("is_mock", False)
        
        self.log(f"Loading video: {video_path}...")
        
        if is_mock or cv2 is None:
            if cv2 is None:
                self.log("OpenCV (cv2) is not available. Falling back to synthetic frame generation.", level=30)
            else:
                self.log("Mock mode enabled. Generating synthetic frames.", level=20)
            
            T = 150
            frames = np.random.rand(T, 112, 112).astype(np.float32)
            state.set("echo_video", frames)
            
            metadata = {
                "patient_id": state.get("patient_id"),
                "frame_count": T,
                "fps": 50,
                "height": 112,
                "width": 112,
                "snr": 15.4
            }
            state.set("metadata", metadata)
            return state

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        frames = []
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 50
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            enhanced = clahe.apply(gray)
            resized = cv2.resize(enhanced, (112, 112))
            normalized = resized.astype(np.float32) / 255.0
            frames.append(normalized)
            
        cap.release()
        
        if not frames:
            raise ValueError(f"Video file {video_path} has 0 frames or is corrupt.")
            
        echo_video = np.array(frames, dtype=np.float32)
        T = len(echo_video)
        
        mean_val = np.mean(echo_video)
        std_val = np.std(echo_video)
        snr = float(mean_val / std_val) if std_val > 0 else 0.0
        
        self.log(f"Successfully loaded {T} frames from {video_path} (FPS={fps}, SNR={snr:.2f}).")
        
        state.set("echo_video", echo_video)
        state.set("metadata", {
            "patient_id": state.get("patient_id"),
            "frame_count": T,
            "fps": fps,
            "height": height,
            "width": width,
            "snr": snr
        })
        
        return state
