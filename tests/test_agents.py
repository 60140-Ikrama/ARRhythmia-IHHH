import os
import unittest
import numpy as np
from utils.state_manager import SharedMemory
from agents.project_manager_agent import ProjectManagerAgent
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

class TestAgents(unittest.TestCase):
    def setUp(self):
        self.pm = ProjectManagerAgent()
        self.state = self.pm.initialize_state("data/synthetic_normal.avi", "PAT_TEST")
        self.state["is_mock"] = True
        self.state["mock_rhythm"] = "normal"
        
    def test_pipeline_execution(self):
        # 1. Data loading
        da = DataAgent()
        da.execute(self.state)
        self.assertIsNotNone(self.state.get("echo_video"))
        
        # 2. Quality
        dq = DataQualityAgent()
        dq.execute(self.state)
        self.assertTrue(self.state.get("quality_metrics")["passed"])
        
        # 3. View
        vc = ViewClassificationAgent()
        vc.execute(self.state)
        self.assertEqual(self.state.get("view_classification")["predicted_view"], "Apical Four Chamber")
        
        # 4. Segmentation
        seg = MultiChamberSegmentationAgent()
        seg.execute(self.state)
        self.assertIn("LV", self.state.get_nested("segmentations", "volumes"))
        
        # 5. Cycle
        cc = CardiacCycleAgent()
        cc.execute(self.state)
        self.assertGreater(self.state.get("heart_rate_bpm"), 0)
        
        # 6. Motion
        mot = CardiacMotionTrackingAgent()
        mot.execute(self.state)
        self.assertGreater(self.state.get("dyssynchrony_index_ms"), 0)
        
        # 7. Strain
        strn = StrainAnalysisAgent()
        strn.execute(self.state)
        self.assertIn("GLS", self.state.get_nested("strain_analysis", "peak_strain"))
        
        # 8. Rhythm
        rhy = MechanicalRhythmAgent()
        rhy.execute(self.state)
        self.assertIn("sdnn_ms", self.state.get("mrvm_features"))
        
        # 9. Pseudo ECG
        ecg = PseudoECGGenerationAgent()
        ecg.execute(self.state)
        self.assertGreater(len(self.state.get_nested("pseudo_ecg", "synthetic_signal")), 0)
        
        # 10. Deep representer
        deep = DeepVideoRepresentationAgent()
        deep.execute(self.state)
        self.assertEqual(len(self.state.get("deep_embeddings")), 128)
        
        # 11. Features
        feat = FeatureEngineeringAgent()
        feat.execute(self.state)
        self.assertEqual(len(self.state.get("fused_feature_vector")), 15)
        
        # 12. Arrhythmia classification
        arr = ArrhythmiaClassificationAgent()
        arr.execute(self.state)
        self.assertEqual(self.state.get("rhythm"), "normal_sinus_rhythm")
        
        # 13. Uncertainty
        unc = UncertaintyAgent()
        unc.execute(self.state)
        self.assertFalse(self.state.get("requires_review"))
        
        # 14. KG
        kg = ClinicalKnowledgeGraphAgent()
        kg.execute(self.state)
        self.assertIn("nodes", self.state.get("kg_subgraph"))
        
        # 15. Reasoning
        reas = ClinicalReasoningAgent()
        reas.execute(self.state)
        self.assertGreater(len(self.state.get_nested("reasoning", "impression")), 0)
        
        # 16. Explainability
        exp = ExplainabilityAgent()
        exp.execute(self.state)
        self.assertEqual(len(self.state.get_nested("explainability_artifacts", "shap_values")), 15)
        
        # 17. Report
        rep = ReportGenerationAgent()
        rep.execute(self.state)
        self.assertTrue(os.path.exists(self.state.get_nested("report_paths", "txt")))

if __name__ == "__main__":
    unittest.main()
