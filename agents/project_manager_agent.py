import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import traceback
from agents.base_agent import BaseAgent


class ProjectManagerAgent(BaseAgent):
    def __init__(self):
        super().__init__("ProjectManagerAgent", "Orchestrator and Validator")

        # Define allowed formats (more realistic for medical pipelines)
        self.allowed_formats = {".avi", ".mp4"}

        # Centralized critical agent definition (avoid hardcoding later)
        self.critical_agents = {
            "data",
            "segmentation",
            "cardiac_cycle",
            "features",
            "arrhythmia"
        }

        # Pipeline is now persistent (not recreated every run)
        self.pipeline = None

    # -----------------------------
    # INPUT VALIDATION
    # -----------------------------
    def validate_input(self, video_path: str, is_mock: bool = False) -> bool:
        """Validates input video path and format."""

        self.log(f"Validating input: {video_path}")

        if is_mock:
            self.log("Mock mode enabled: skipping file validation.")
            return True

        if not video_path:
            self.log("Validation failed: empty video path", level=40)
            return False

        if not os.path.exists(video_path):
            self.log(f"Validation failed: file not found -> {video_path}", level=40)
            return False

        ext = os.path.splitext(video_path)[-1].lower()
        if ext not in self.allowed_formats:
            self.log(
                f"Validation failed: unsupported format {ext}. Allowed: {self.allowed_formats}",
                level=40
            )
            return False

        return True

    # -----------------------------
    # STATE INITIALIZATION
    # -----------------------------
    def initialize_state(self, video_path: str, patient_id: str) -> dict:
        """Creates shared pipeline state."""

        self.log(f"Initializing pipeline state for patient {patient_id}")

        return {
            "video_path": video_path,
            "patient_id": patient_id,
            "is_mock": False,

            # Execution tracking
            "pipeline_status": [],
            "errors": [],
            "timings": {},

            # Metadata
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),

            # Core outputs
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

    # -----------------------------
    # RETRY WRAPPER
    # -----------------------------
    def execute_agent_with_retry(self, agent_instance, state: dict, max_retries: int = 3) -> bool:
        """Executes agent with retry + logging + timing."""

        agent_name = agent_instance.name
        self.log(f"Routing -> {agent_name}")

        for attempt in range(1, max_retries + 1):
            start_time = time.time()

            try:
                agent_instance.update_status(state, "RUNNING")
                agent_instance.execute(state)
                agent_instance.update_status(state, "COMPLETE")

                duration = time.time() - start_time
                state["timings"][agent_name] = duration

                self.log(f"{agent_name} completed in {duration:.2f}s (attempt {attempt})")
                return True

            except Exception as e:
                duration = time.time() - start_time

                error_trace = traceback.format_exc()

                self.log(
                    f"{agent_name} failed (attempt {attempt}/{max_retries})\n{error_trace}",
                    level=40
                )

                state["errors"].append({
                    "agent": agent_name,
                    "attempt": attempt,
                    "error": str(e),
                    "trace": error_trace
                })

                if attempt == max_retries:
                    agent_instance.update_status(state, "FAILED")
                    agent_instance.add_error(
                        state,
                        f"{agent_name} failed after {max_retries} retries"
                    )
                    return False

                time.sleep(0.5 * attempt)  # exponential backoff

        return False

    # -----------------------------
    # PIPELINE EXECUTION
    # -----------------------------
    def execute(self, state: dict) -> dict:
        """Runs full multi-agent pipeline."""

        self.update_status(state, "RUNNING")

        is_mock = state.get("is_mock", False)

        # Validate input
        if not self.validate_input(state.get("video_path"), is_mock):
            self.add_error(state, "Input validation failed")
            self.update_status(state, "FAILED")
            state["requires_review"] = True
            return state

        # Lazy imports (avoid circular dependency)
        from agents.data_agent import DataAgent
        from agents.segmentation_agent import SegmentationAgent
        from agents.cardiac_cycle_agent import CardiacCycleAgent
        from agents.hrv_agent import MechanicalHRVAgent
        from agents.motion_agent import MotionAnalysisAgent
        from agents.feature_agent import FeatureEngineeringAgent
        from agents.arrhythmia_agent import ArrhythmiaDetectionAgent
        from agents.report_agent import ClinicalReportAgent

        # Initialize pipeline once per run
        if self.pipeline is None:
            self.pipeline = [
                ("data", DataAgent()),
                ("segmentation", SegmentationAgent()),
                ("cardiac_cycle", CardiacCycleAgent()),
                ("hrv", MechanicalHRVAgent()),
                ("motion", MotionAnalysisAgent()),
                ("features", FeatureEngineeringAgent()),
                ("arrhythmia", ArrhythmiaDetectionAgent()),
                ("report", ClinicalReportAgent())
            ]

        # Run pipeline
        for step_name, agent in self.pipeline:

            success = self.execute_agent_with_retry(agent, state)

            if not success:

                if step_name in self.critical_agents:
                    self.log(
                        f"Critical failure at {step_name}. Aborting pipeline.",
                        level=40
                    )
                    state["requires_review"] = True
                    self.update_status(state, "FAILED")
                    return state

                else:
                    self.log(
                        f"Non-critical agent failed: {step_name}. Continuing pipeline.",
                        level=30
                    )

        # -----------------------------
        # FINAL VALIDATION
        # -----------------------------
        self.log("Running final validation checks...")

        confidence = state.get("confidence", 0.0)

        # More flexible threshold system
        threshold = state.get("confidence_threshold", 0.70)

        if confidence < threshold:
            self.log(
                f"Low confidence ({confidence:.2f} < {threshold}). Flagging review.",
                level=30
            )
            state["requires_review"] = True

        if state.get("errors"):
            self.log("Pipeline completed with errors -> review required", level=30)
            state["requires_review"] = True

        self.update_status(state, "COMPLETE")

        self.log("Pipeline execution finished successfully.")
        return state


if __name__ == "__main__":
    print("ProjectManagerAgent module initialized successfully.")
    agent = ProjectManagerAgent()
    print(f"Agent Name: {agent.name}")
    print(f"Agent Role: {agent.role}")
