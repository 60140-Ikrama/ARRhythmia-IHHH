import os
import time
from agents.base_agent import BaseAgent

class ProjectManagerAgent(BaseAgent):
    def __init__(self):
        super().__init__("ProjectManagerAgent", "Orchestrator and Validator")

    def validate_input(self, video_path: str, is_mock: bool = False) -> bool:
        """Validates the input video path and format."""
        self.log(f"Validating input: {video_path}")
        
        # If running mock, we can skip file checks since we'll generate mock data
        if is_mock:
            self.log("Mock execution mode: skipping physical video validation.")
            return True
            
        if not os.path.exists(video_path):
            self.log(f"Validation failed: File does not exist at {video_path}", level=40) # ERROR
            return False
            
        if not video_path.lower().endswith(".avi"):
            self.log(f"Validation failed: Video format must be .avi, got {video_path}", level=40)
            return False
            
        return True

    def initialize_state(self, video_path: str, patient_id: str) -> dict:
        """Initializes the shared AgentState dictionary."""
        self.log(f"Initializing state for patient {patient_id}")
        return {
            "video_path": video_path,
            "patient_id": patient_id,
            "pipeline_status": [],
            "errors": [],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            
            # Placeholder outputs for agents
            "echo_video": None,
            "metadata": {},
            "segmentation_masks": None,
            "volume_curve": None,
            "segmentation_confidence": 0.0,
            "es_frames": [],
            "ed_frames": [],
            "rr_intervals_sec": [],
            "heart_rate_bpm": 0.0,
            "ejection_fraction": 0.0,
            "irregularity_index": 0.0,
            "hrv_time": {},
            "hrv_freq": {},
            "hrv_nonlinear": {},
            "dyssynchrony_index_ms": 0.0,
            "motion_irregularity": 0.0,
            "motion_pattern": "unknown",
            "feature_vector": None,
            "feature_names": [],
            "rhythm": "unknown",
            "confidence": 0.0,
            "probabilities": {},
            "evidence": [],
            "requires_review": False,
            "narrative_report": "",
            "structured_json": {}
        }

    def execute_agent_with_retry(self, agent_instance, state: dict, max_retries: int = 3) -> bool:
        """Runs an agent with retry logic."""
        agent_name = agent_instance.name
        self.log(f"Routing task to {agent_name}...")
        
        for attempt in range(1, max_retries + 1):
            try:
                agent_instance.update_status(state, "RUNNING")
                agent_instance.execute(state)
                agent_instance.update_status(state, "COMPLETE")
                self.log(f"{agent_name} executed successfully on attempt {attempt}.")
                return True
            except Exception as e:
                self.log(f"Attempt {attempt} failed for {agent_name}. Error: {str(e)}", level=30) # WARNING
                if attempt == max_retries:
                    agent_instance.update_status(state, "FAILED")
                    agent_instance.add_error(state, f"Failed after {max_retries} retries. Error: {str(e)}")
                    return False
                time.sleep(0.5)
        return False

    def execute(self, state: dict) -> dict:
        """Orchestrates the entire 9-agent pipeline."""
        self.update_status(state, "RUNNING")
        
        # 1. Validation
        is_mock = state.get("is_mock", False)
        if not self.validate_input(state["video_path"], is_mock):
            self.add_error(state, "Input video validation failed.")
            self.update_status(state, "FAILED")
            return state

        # Lazy import of other agents to prevent circular dependency / pre-mature loading
        from agents.data_agent import DataAgent
        from agents.segmentation_agent import SegmentationAgent
        from agents.cardiac_cycle_agent import CardiacCycleAgent
        from agents.hrv_agent import MechanicalHRVAgent
        from agents.motion_agent import MotionAnalysisAgent
        from agents.feature_agent import FeatureEngineeringAgent
        from agents.arrhythmia_agent import ArrhythmiaDetectionAgent
        from agents.report_agent import ClinicalReportAgent

        # Pipeline execution steps
        pipeline = [
            ("data", DataAgent()),
            ("segmentation", SegmentationAgent()),
            ("cardiac_cycle", CardiacCycleAgent()),
            ("hrv", MechanicalHRVAgent()),
            ("motion", MotionAnalysisAgent()),
            ("features", FeatureEngineeringAgent()),
            ("arrhythmia", ArrhythmiaDetectionAgent()),
            ("report", ClinicalReportAgent())
        ]

        for step_name, agent in pipeline:
            success = self.execute_agent_with_retry(agent, state)
            
            if not success:
                # Critical agents check
                if step_name in ["data", "segmentation", "cardiac_cycle", "features", "arrhythmia"]:
                    self.log(f"Critical agent {agent.name} failed. Aborting pipeline.", level=40)
                    self.update_status(state, "FAILED")
                    state["requires_review"] = True
                    return state
                else:
                    self.log(f"Non-critical agent {agent.name} failed. Continuing with caution.", level=30)

        # Final check
        self.log("Pipeline processing completed. Conducting final verification...")
        
        # Flag if arrhythmia confidence < 70% or if there are errors
        confidence = state.get("confidence", 0.0)
        if confidence < 0.70:
            self.log(f"Confidence score {confidence:.2f} is below threshold 0.70. Flagging for review.", level=30)
            state["requires_review"] = True
        
        if len(state.get("errors", [])) > 0:
            self.log("Pipeline completed with errors. Flagging for human review.", level=30)
            state["requires_review"] = True

        self.update_status(state, "COMPLETE")
        return state
