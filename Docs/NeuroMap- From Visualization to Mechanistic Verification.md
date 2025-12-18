# **NeuroMap: From Visualization to Mechanistic Verification**

This document outlines the strategic pivot for the NeuroMap repository to align with the standards of professional AI safety and Mechanistic Interpretability research.

## **1\. Strategic Framework (Grounded in default\_600k)**

### **1.1 Shift from Stage 2 (Exploration) to Stage 3 (Understanding)**

Currently, NeuroMap excels at **Gaining Surface Area** (Neel Nanda's Stage 2). We have extraction, clustering, and graph plotting. To be research-ready, we must move to **Stage 3: Testing Hypotheses**.

* **Existing Hypothesis:** "The model organizes modular arithmetic concepts in a circular topology."  
* **Required Verification:** We must prove that this circular topology is *causally responsible* for the model's output.

### **1.2 Prioritizing Information Rate (Steinhardt's Strategy)**

Instead of adding "more" features (like more graph layouts), we will prioritize **De-risking** the interpretation.

* **The Risk:** The circular graph might be a "hallucination" of the layout algorithm or a side effect of the embedding layer that the rest of the model ignores.  
* **The Solution:** Implement **Causal Scrubbing** and **Path Patching** to verify if removing these "circular" features actually breaks the model's logic.

## **2\. Technical Implementation Roadmap**

### **Phase 1: The "Faithfulness" Upgrade**

We will implement a standard metric for how well our extracted graph represents the model.

* **Metric:** *Faithfulness Score*. If we replace the model's activations with the "concept centers" from our ClusteringExtractor, what is the drop in accuracy?  
* **Implementation:** Add a ConceptReconstruction method to the analysis pipeline.

### **Phase 2: Causal Intervention Module**

The "Mecha-Interp" community relies on interventions.

* **Activation Patching:** Swap activations between different inputs (e.g., $a+b \\pmod{p}$ vs $c+d \\pmod{p}$) at the "Concept Node" level.  
* **Saliency Mapping:** Use gradients to show which specific neurons in the MLP layers contribute most to the "Circular" structure identified by NeuroMap.

### **Phase 3: SAE Rigor**

The current SAE implementation is basic. Professional research requires:

* **Gated SAEs:** To solve the "shrinkage" problem where L1 penalties reduce the magnitude of activations.  
* **Feature Buffers:** Using a larger "shuffle buffer" for SAE training to ensure features are truly disentangled.

## **3\. Communication & Distillation (Stage 4\)**

To be useful to researchers, NeuroMap should produce "Distillations":

* **Interoperability:** Support for TransformerLens. Researchers should be able to load a HookedTransformer and run NeuroMap diagnostics instantly.  
* **The Narrative:** Instead of a tool overview, the README should present a **case study**: "Recovering the Fourier Algorithm in a 2-Layer Transformer using NeuroMap."

## **4\. Specific Code Improvements (Immediate Actions)**

1. **Refactor analysis/concept\_extractors.py**: Add a LinearRepresentation class to check the **Linear Representation Hypothesis** (Are concepts stored as directions in space?).  
2. **Update graph/concept\_graph.py**: Add a "Causal Edge" weight, where edge weight \= the impact of patching one node's activation into another.