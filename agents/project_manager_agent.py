import os
import json
import time
from agents.base_agent import BaseAgent
from utils.state_manager import SharedMemory

class ProjectManagerAgent(BaseAgent):
    def __init__(self):
        super().__init__("ProjectManagerAgent", "Orchestrator and Validator")

    def validate_input(self, video_path: str, is_mock: bool = False) -> bool:
        """Validates the input video path and format."""
        self.log(f"Validating input: {video_path}")
        
        if is_mock:
            self.log("Mock execution mode: skipping physical video validation.")
            return True
            
        if not os.path.exists(video_path):
            self.log(f"Validation failed: File does not exist at {video_path}", level=40)
            return False
            
        if not video_path.lower().endswith(".avi"):
            self.log(f"Validation failed: Video format must be .avi, got {video_path}", level=40)
            return False
            
        return True

    def initialize_state(self, video_path: str, patient_id: str) -> SharedMemory:
        """Initializes the shared SharedMemory state."""
        self.log(f"Initializing SharedMemory state for patient {patient_id}")
        state = SharedMemory(patient_id, video_path)
        
        # Load configuration file
        config_path = "configs/pipeline_config.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    configs = json.load(f)
                state.set("configs", configs)
                self.log("Pipeline configs loaded successfully.")
            except Exception as e:
                self.log(f"Could not load configs: {str(e)}. Using defaults.", level=30)
                
        return state

    def execute_agent_with_retry(self, agent_instance, state: SharedMemory, max_retries: int = 3) -> bool:
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
                self.log(f"Attempt {attempt} failed for {agent_name}. Error: {str(e)}", level=30)
                if attempt == max_retries:
                    agent_instance.update_status(state, "FAILED")
                    agent_instance.add_error(state, f"Failed after {max_retries} retries. Error: {str(e)}")
                    return False
                time.sleep(0.2)
        return False

    def execute(self, state: SharedMemory) -> SharedMemory:
        """Orchestrates the entire 16-agent pipeline."""
        self.update_status(state, "RUNNING")
        
        is_mock = state.get("is_mock", False)
        if not self.validate_input(state.get("video_path"), is_mock):
            self.add_error(state, "Input video validation failed.")
            self.update_status(state, "FAILED")
            return state

        # Lazy imports of agents to prevent circular imports
        from agents.data_agent import DataAgent
        from agents.data_quality_agent import DataQualityAgent
        from agents.view_classification_agent import ViewClassificationAgent
        from agents.segmentation_agent import MultiChamberSegmentationAgent
        from agents.cardiac_cycle_agent import CardiacCycleAgent
        from agents.motion_agent import CardiacMotionTrackingAgent
        from agents.strain_agent import StrainAnalysisAgent
        from agents.rhythm_agent import MechanicalRhythmAgent
        from agents.pseudo_ecg_agent import PseudoECGGenerationAgent
        from agents.deep_video_agent import DeepVideoRepresentationAgent
        from agents.feature_agent import FeatureEngineeringAgent
        from agents.arrhythmia_agent import ArrhythmiaClassificationAgent
        from agents.uncertainty_agent import UncertaintyAgent
        from agents.knowledge_graph_agent import ClinicalKnowledgeGraphAgent
        from agents.clinical_reasoning_agent import ClinicalReasoningAgent
        from agents.explainability_agent import ExplainabilityAgent
        from agents.report_agent import ReportGenerationAgent

        # Pipeline execution steps
        pipeline = [
            ("data", DataAgent()),
            ("quality", DataQualityAgent()),
            ("view", ViewClassificationAgent()),
            ("segmentation", MultiChamberSegmentationAgent()),
            ("cardiac_cycle", CardiacCycleAgent()),
            ("motion", CardiacMotionTrackingAgent()),
            ("strain", StrainAnalysisAgent()),
            ("rhythm", MechanicalRhythmAgent()),
            ("pseudo_ecg", PseudoECGGenerationAgent()),
            ("deep_video", DeepVideoRepresentationAgent()),
            ("features", FeatureEngineeringAgent()),
            ("arrhythmia", ArrhythmiaClassificationAgent()),
            ("uncertainty", UncertaintyAgent()),
            ("kg", ClinicalKnowledgeGraphAgent()),
            ("reasoning", ClinicalReasoningAgent()),
            ("explainability", ExplainabilityAgent()),
            ("report", ReportGenerationAgent())
        ]

        for step_name, agent in pipeline:
            success = self.execute_agent_with_retry(agent, state)
            
            if not success:
                # Critical agents check
                critical_steps = ["data", "quality", "view", "segmentation", "cardiac_cycle", "features", "arrhythmia", "reasoning", "report"]
                if step_name in critical_steps:
                    self.log(f"Critical agent {agent.name} failed. Aborting pipeline.", level=40)
                    self.update_status(state, "FAILED")
                    state.set("requires_review", True)
                    return state
                else:
                    self.log(f"Non-critical agent {agent.name} failed. Continuing with caution.", level=30)

        # Final verification
        self.log("Pipeline processing completed. Conducting final verification...")
        
        # Check review flag
        requires_review = state.get("requires_review", False)
        if len(state.get("errors", [])) > 0:
            self.log("Pipeline completed with errors. Flagging for human review.", level=30)
            state.set("requires_review", True)

        self.update_status(state, "COMPLETE")
        return state
