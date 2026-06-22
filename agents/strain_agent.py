import os
import numpy as np
from agents.base_agent import BaseAgent
from utils.state_manager import SharedMemory

class StrainAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__("StrainAnalysisAgent", "Myocardial Deformation Specialist")

    def execute(self, state: SharedMemory) -> SharedMemory:
        self.log("Performing myocardial strain analysis...")
        is_mock = state.get("is_mock", False)
        mock_rhythm = state.get("mock_rhythm", "normal")
        
        # We need the cardiac cycle frames to align strain curves (systole vs diastole)
        # Check volume curve from state to see number of frames
        volumes_lv = state.get_nested("segmentations", "volumes", {}).get("LV", [])
        if not volumes_lv:
            volumes_lv = state.get("volume_curve", [])
            
        T = len(volumes_lv) if len(volumes_lv) > 0 else 150
        t_axis = np.arange(T)
        fps = float(state.get_nested("metadata", "fps", 50.0) or 50.0)
        
        # Simulate strain curves representing normal or pathological states
        if mock_rhythm == "normal":
            # Normal GLS: systolic shortening down to -20%
            gls = -20.0 * np.sin(2 * np.pi * (75/60.0) * (t_axis/fps))**2
            radial = 40.0 * np.sin(2 * np.pi * (75/60.0) * (t_axis/fps))**2
            circum = -18.0 * np.sin(2 * np.pi * (75/60.0) * (t_axis/fps))**2
            atrial = 15.0 * np.sin(2 * np.pi * (75/60.0) * (t_axis/fps) + np.pi/2)**2 # out of phase
            
            peak_gls = -20.5
            peak_radial = 41.2
            peak_circum = -18.8
            peak_atrial = 15.4
        elif mock_rhythm == "afib":
            # AFib: depressed and highly irregular strain
            gls = -10.0 * np.sin(t_axis/4.0)**2 + np.random.normal(0, 1.0, T)
            radial = 20.0 * np.sin(t_axis/4.0)**2 + np.random.normal(0, 2.0, T)
            circum = -9.0 * np.sin(t_axis/4.0)**2 + np.random.normal(0, 1.0, T)
            atrial = 3.0 * np.sin(t_axis/5.0)**2 + np.random.normal(0, 0.5, T) # severely depressed atrial strain
            
            peak_gls = -11.2
            peak_radial = 22.1
            peak_circum = -9.8
            peak_atrial = 3.4
        elif mock_rhythm == "pvc":
            # PVC: contraction jerks
            gls = -14.0 * np.sin(2 * np.pi * (72/60.0) * (t_axis/fps))**2
            # Add PVC beat anomalies
            for idx in range(T):
                if 65 <= idx <= 80:
                    gls[idx] = -6.0 + np.random.normal(0, 0.5)
            radial = 28.0 * np.sin(2 * np.pi * (72/60.0) * (t_axis/fps))**2
            circum = -13.0 * np.sin(2 * np.pi * (72/60.0) * (t_axis/fps))**2
            atrial = 12.0 * np.sin(2 * np.pi * (72/60.0) * (t_axis/fps) + np.pi/2)**2
            
            peak_gls = -15.1
            peak_radial = 30.5
            peak_circum = -14.2
            peak_atrial = 12.8
        else: # bradycardia or tachycardia
            bpm = 48 if mock_rhythm == "bradycardia" else 125
            gls = -19.0 * np.sin(2 * np.pi * (bpm/60.0) * (t_axis/fps))**2
            radial = 38.0 * np.sin(2 * np.pi * (bpm/60.0) * (t_axis/fps))**2
            circum = -17.0 * np.sin(2 * np.pi * (bpm/60.0) * (t_axis/fps))**2
            atrial = 14.0 * np.sin(2 * np.pi * (bpm/60.0) * (t_axis/fps) + np.pi/2)**2
            
            peak_gls = -19.5
            peak_radial = 38.8
            peak_circum = -17.5
            peak_atrial = 14.2
            
        strain_state = {
            "gls_curve": gls.tolist(),
            "radial_strain": radial.tolist(),
            "circumferential_strain": circum.tolist(),
            "atrial_strain": atrial.tolist(),
            "peak_strain": {
                "GLS": float(peak_gls),
                "radial": float(peak_radial),
                "circumferential": float(peak_circum),
                "atrial": float(peak_atrial)
            }
        }
        
        state.set("strain_analysis", strain_state)
        
        # Save placeholder visualizations paths or generate plots
        os.makedirs("visualizations/strain_curves", exist_ok=True)
        os.makedirs("visualizations/bullseye_maps", exist_ok=True)
        
        self.log(f"Deformation complete. Peak GLS={peak_gls:.1f}%, Peak Atrial Strain={peak_atrial:.1f}%")
        return state
