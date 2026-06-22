from agents.rhythm_agent import MechanicalRhythmAgent

class MechanicalHRVAgent(MechanicalRhythmAgent):
    """
    Subclass of MechanicalRhythmAgent for backward compatibility.
    """
    def __init__(self):
        super().__init__()
        self.name = "MechanicalHRVAgent"
        self.role = "Mechanical HRV Analysis (Legacy Mapper)"
