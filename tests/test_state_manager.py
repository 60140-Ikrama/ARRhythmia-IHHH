import unittest
import numpy as np
from utils.state_manager import SharedMemory

class TestStateManager(unittest.TestCase):
    def test_state_initialization(self):
        state = SharedMemory("PAT_TEST", "mock_video.avi")
        self.assertEqual(state["patient_id"], "PAT_TEST")
        self.assertEqual(state["video_path"], "mock_video.avi")
        self.assertEqual(len(state["pipeline_status"]), 0)
        
    def test_dict_compatibility(self):
        state = SharedMemory("PAT_TEST", "mock_video.avi")
        state["is_mock"] = True
        self.assertTrue(state["is_mock"])
        
        # Test default legacy mapping
        state.update_nested("segmentations", "ejection_fraction", 62.5)
        self.assertEqual(state["ejection_fraction"], 62.5)
        
    def test_transaction_logging(self):
        state = SharedMemory("PAT_TEST", "mock_video.avi")
        state.log_pipeline_status("TestAgent", "COMPLETE")
        self.assertIn("TestAgent: COMPLETE", state["pipeline_status"])

if __name__ == "__main__":
    unittest.main()
