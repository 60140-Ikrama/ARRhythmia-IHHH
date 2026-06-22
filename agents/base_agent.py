import logging
from utils.state_manager import SharedMemory

class BaseAgent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.logger = logging.getLogger(self.name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def log(self, message: str, level: int = logging.INFO):
        """Standardized logger helper."""
        self.logger.log(level, message)

    def update_status(self, state: SharedMemory, status: str):
        """Updates this agent's status in the shared state's transaction log."""
        if isinstance(state, SharedMemory):
            state.log_pipeline_status(self.name, status)
        else:
            # Fallback for plain dictionary
            if "pipeline_status" not in state:
                state["pipeline_status"] = []
            state["pipeline_status"].append(f"{self.name}: {status}")
            self.log(f"Status changed to {status}")

    def add_error(self, state: SharedMemory, error_msg: str):
        """Appends an error message to the shared state's error list."""
        if isinstance(state, SharedMemory):
            state.add_error(self.name, error_msg)
        else:
            # Fallback for plain dictionary
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(f"[{self.name}] {error_msg}")
            self.log(f"Error: {error_msg}", logging.ERROR)

    def execute(self, state: SharedMemory) -> SharedMemory:
        """
        Core logic of the agent. Modifies the shared state.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement execute(self, state: SharedMemory)")
