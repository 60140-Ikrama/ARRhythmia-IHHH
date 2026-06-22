import networkx as nx

class ClinicalKnowledgeGraph:
    """
     clinical cardiology knowledge graph detailing pathophysiological links 
    between electrical arrhythmias, mechanical changes, and hemodynamic consequences.
    """
    def __init__(self):
        self.g = nx.DiGraph()
        self._build_graph()

    def _build_graph(self):
        # Define arrhythmia nodes
        arrhythmias = ["atrial_fibrillation", "pvc", "bradycardia", "tachycardia", "normal_sinus_rhythm"]
        for r in arrhythmias:
            self.g.add_node(r, type="arrhythmia")
            
        # Define mechanical features nodes
        mech_nodes = [
            ("irregular_contraction", "mechanical_consequence"),
            ("absent_atrial_kick", "mechanical_consequence"),
            ("short_diastolic_filling", "mechanical_consequence"),
            ("compensatory_pause", "mechanical_consequence"),
            ("ectopic_contraction", "mechanical_consequence"),
            ("rapid_filling", "mechanical_consequence"),
            
            # Chamber abnormalities
            ("atrial_enlargement", "chamber_abnormality"),
            ("ventricular_dyssynchrony", "chamber_abnormality"),
            ("systolic_dysfunction", "chamber_abnormality"),
            
            # Motion patterns
            ("chaotic_myocardial_motion", "motion_pattern"),
            ("localized_wall_jerk", "motion_pattern"),
            ("stable_synchronous_motion", "motion_pattern"),
            
            # Strain changes
            ("depressed_global_longitudinal_strain", "strain_change"),
            ("impaired_atrial_strain", "strain_change"),
            ("normal_strain_values", "strain_change")
        ]
        for node, n_type in mech_nodes:
            self.g.add_node(node, type=n_type)
            
        # Add relationships / edges
        # AFib pathways
        self.g.add_edge("atrial_fibrillation", "irregular_contraction", relation="causes")
        self.g.add_edge("atrial_fibrillation", "absent_atrial_kick", relation="causes")
        self.g.add_edge("absent_atrial_kick", "impaired_atrial_strain", relation="manifests_as")
        self.g.add_edge("absent_atrial_kick", "atrial_enlargement", relation="leads_to")
        self.g.add_edge("irregular_contraction", "chaotic_myocardial_motion", relation="manifests_as")
        self.g.add_edge("chaotic_myocardial_motion", "depressed_global_longitudinal_strain", relation="contributes_to")
        self.g.add_edge("atrial_fibrillation", "ventricular_dyssynchrony", relation="causes")
        
        # PVC pathways
        self.g.add_edge("pvc", "ectopic_contraction", relation="causes")
        self.g.add_edge("ectopic_contraction", "localized_wall_jerk", relation="manifests_as")
        self.g.add_edge("localized_wall_jerk", "ventricular_dyssynchrony", relation="induces")
        self.g.add_edge("pvc", "compensatory_pause", relation="followed_by")
        
        # Bradycardia pathways
        self.g.add_edge("bradycardia", "short_diastolic_filling", relation="prevents")
        self.g.add_edge("bradycardia", "stable_synchronous_motion", relation="maintains")
        
        # Tachycardia pathways
        self.g.add_edge("tachycardia", "short_diastolic_filling", relation="causes")
        self.g.add_edge("short_diastolic_filling", "systolic_dysfunction", relation="leads_to")
        
        # Normal pathway
        self.g.add_edge("normal_sinus_rhythm", "stable_synchronous_motion", relation="maintains")
        self.g.add_edge("stable_synchronous_motion", "normal_strain_values", relation="supports")

    def get_pathology_subgraph(self, rhythm: str) -> dict:
        """Retrieves subgraph of pathobiological connections for a specific rhythm."""
        if rhythm not in self.g:
            return {}
            
        # Get successor paths (one-hop and two-hop)
        sub_nodes = {rhythm}
        for s in self.g.successors(rhythm):
            sub_nodes.add(s)
            for s2 in self.g.successors(s):
                sub_nodes.add(s2)
                
        sub = self.g.subgraph(sub_nodes)
        
        # Serialize subgraph for shared state representation
        nodes_info = {n: self.g.nodes[n] for n in sub.nodes}
        edges_info = []
        for u, v in sub.edges:
            edges_info.append({
                "source": u,
                "target": v,
                "relation": self.g.edges[u, v]["relation"]
            })
            
        return {
            "nodes": nodes_info,
            "edges": edges_info
        }
