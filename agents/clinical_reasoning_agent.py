from agents.base_agent import BaseAgent
from utils.state_manager import SharedMemory

class ClinicalReasoningAgent(BaseAgent):
    def __init__(self):
        super().__init__("ClinicalReasoningAgent", "Diagnostic Reasoning Scriptor")

    def get_impression(self, rhythm: str, hr: float, ef: float, confidence: float, evidence: list) -> str:
        """Translates classification findings and features into clinical cardiology reasoning."""
        rhythm_clean = rhythm.replace("_", " ").upper()
        
        base_impression = f"Diagnostic Impression favors {rhythm_clean} ({confidence:.1f}% confidence) with LVEF of {ef:.1f}% and Heart Rate of {hr:.1f} BPM. "
        
        if "atrial fibrillation" in rhythm.lower():
            reasoning = (
                f"Pathophysiologically, the echocardiographic volumes exhibit high cycle-to-cycle irregularity "
                f"(Irregularity Index > 0.15), coupled with depressed myocardial shortening and a significant loss "
                f"of coordinates representing normal atrial filling phases. This mechanical disruption indicates "
                f"an absent active 'atrial kick' contribution (impaired atrial strain), which is the classic mechanical signature "
                f"of atrial fibrillation. Dense motion tracking shows chaotic, non-synchronous myocardial displacement."
            )
        elif "pvc" in rhythm.lower():
            reasoning = (
                f"Motion parameters reveal localized myocardial wall motion dyssynchrony and a sudden, abrupt wall velocity "
                f"jerk signature. This indicates an ectopic ventricular depolarization causing a premature contraction prior "
                f"to normal atrial filling, followed by a compensatory diastolic pause. This pattern matches mechanical PVC manifestations."
            )
        elif "bradycardia" in rhythm.lower():
            reasoning = (
                f"The mechanical contraction rate is severely depressed ({hr:.1f} BPM), but overall rhythm regularity "
                f"and wall displacement indices remain within normal physiological patterns. No active dyssynchrony is present."
            )
        elif "tachycardia" in rhythm.lower():
            reasoning = (
                f"The contraction rate is elevated ({hr:.1f} BPM), leading to abbreviated diastolic filling periods "
                f"and a resulting reduction in end-diastolic volume. Coordinate contraction remains regular and synchronous."
            )
        else:
            reasoning = "Normal cardiac cycle dynamics, synchronous contraction patterns, and stable mechanical rhythm indices."
            
        return base_impression + "\n\nPathobiology Reasoning:\n" + reasoning

    def get_recommendations(self, rhythm: str, ef: float, confidence: float) -> list:
        """Cardiology clinical recommendations based on guidelines."""
        recs = []
        rhythm_lower = rhythm.lower()
        
        if "normal_sinus_rhythm" in rhythm_lower:
            recs.append("Routine clinical follow-up as indicated.")
        elif "atrial_fibrillation" in rhythm_lower:
            recs.append("Urgent 12-lead ECG to confirm Atrial Fibrillation.")
            recs.append("Evaluate stroke risk profile (CHA2DS2-VASc score) and anticoagulation needs.")
            recs.append("Refer to Cardiology / Electrophysiology for rate vs rhythm control strategy.")
        elif "pvc" in rhythm_lower:
            recs.append("24-hour Holter monitoring to quantify PVC burden.")
            recs.append("Check serum electrolytes (potassium, magnesium) and thyroid panels.")
            recs.append("Advise patient to limit cardiovascular stimulants (caffeine, nicotine).")
        elif "bradycardia" in rhythm_lower:
            recs.append("12-lead ECG to rule out high-degree AV blocks.")
            recs.append("Review medications for chronotropic agents (beta-blockers, calcium channel blockers).")
        elif "tachycardia" in rhythm_lower:
            recs.append("ECG correlation to identify narrow vs wide complex tachycardia.")
            recs.append("Screen for underlying triggers (thyrotoxicosis, infection, volume depletion).")
            
        if confidence < 80.0:
            recs.append("Low diagnostic confidence: correlate findings with manual trace or cardiac telemetry.")
        if ef < 40.0:
            recs.append("Reduced LVEF (<40%): Consider Guideline-Directed Medical Therapy (GDMT) for heart failure.")
            
        return recs

    def execute(self, state: SharedMemory) -> SharedMemory:
        self.log("Synthesizing clinical diagnostic reasoning...")
        
        preds = state.get("predictions", {})
        rhythm = preds.get("rhythm", "unknown")
        confidence = state.get("confidence", 0.0) * 100.0
        
        # Read metrics from state
        ef = state.get("ejection_fraction", 0.0)
        mrvm = state.get("mrvm_features", {})
        hr = mrvm.get("heart_rate_bpm", 75.0)
        evidence = state.get("evidence", [])
        
        # Build explanation
        impression = self.get_impression(rhythm, hr, ef, confidence, evidence)
        recommendations = self.get_recommendations(rhythm, ef, confidence)
        
        reasoning_state = {
            "impression": impression,
            "evidence_points": evidence,
            "recommendations": recommendations
        }
        
        state.set("reasoning", reasoning_state)
        # Update legacy keys
        state.set("evidence", evidence)
        
        self.log("Clinical reasoning compiled successfully.")
        return state
