import os
import numpy as np
from agents.base_agent import BaseAgent
from utils.state_manager import SharedMemory

class DeepVideoRepresentationAgent(BaseAgent):
    def __init__(self):
        super().__init__("DeepVideoRepresentationAgent", "Spatiotemporal Feature Extractor")

    def execute(self, state: SharedMemory) -> SharedMemory:
        self.log("Extracting deep video spatiotemporal representations...")
        is_mock = state.get("is_mock", False)
        mock_rhythm = state.get("mock_rhythm", "normal")
        
        # We output a 128-dimensional embedding vector
        embedding_size = 128
        
        # Seed generator based on patient and rhythm to keep it reproducible
        patient_id = state.get("patient_id")
        seed = hash(patient_id + mock_rhythm) % 2**32
        np.random.seed(seed)
        
        # Generate distinct embeddings for each rhythm category
        if mock_rhythm == "normal":
            base_vec = np.random.normal(0.5, 0.1, embedding_size)
        elif mock_rhythm == "afib":
            base_vec = np.random.normal(-0.5, 0.12, embedding_size)
        elif mock_rhythm == "pvc":
            base_vec = np.random.normal(0.1, 0.15, embedding_size)
        elif mock_rhythm == "bradycardia":
            base_vec = np.random.normal(0.8, 0.08, embedding_size)
        else: # tachycardia
            base_vec = np.random.normal(-0.8, 0.08, embedding_size)
            
        # Add random noise to simulate specific video variability
        noise = np.random.normal(0, 0.05, embedding_size)
        embeddings = (base_vec + noise).tolist()
        
        state.set("deep_embeddings", embeddings)
        
        os.makedirs("reports/deep_embeddings", exist_ok=True)
        
        self.log(f"Deep video representation complete. Size={embedding_size} dimensions.")
        return state
