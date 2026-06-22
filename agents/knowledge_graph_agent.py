from agents.base_agent import BaseAgent
from utils.state_manager import SharedMemory
from utils.clinical_kg import ClinicalKnowledgeGraph

class ClinicalKnowledgeGraphAgent(BaseAgent):
    def __init__(self):
        super().__init__("ClinicalKnowledgeGraphAgent", "Clinical Pathobiology Graph Linker")
        self.kg = ClinicalKnowledgeGraph()

    def execute(self, state: SharedMemory) -> SharedMemory:
        self.log("Querying clinical knowledge graph for pathophysiological context...")
        
        preds = state.get("predictions", {})
        rhythm = preds.get("rhythm", "unknown")
        
        # Query pathobiology links for predicted rhythm
        subgraph = self.kg.get_pathology_subgraph(rhythm)
        
        state.set("kg_subgraph", subgraph)
        
        self.log(f"Linked diagnostic subgraph containing {len(subgraph.get('nodes', {}))} clinical nodes.")
        return state
