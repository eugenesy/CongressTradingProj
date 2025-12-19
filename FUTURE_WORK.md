# Future Work & Handover Plan

This document outlines the next steps for Project Chocolate, as suggested by the research team and expert feedback. It serves as a roadmap for future development.

## 1. Priority: Baseline Model Comparison
**Goal**: Quantify the value-add of the Graph Network by comparing it against a strong non-graph baseline.
**Assignee**: Teammate

*   **Task**: Implement a Non-Graph MLP Baseline.
*   **Methodology**:
    *   Use the exact same input features as the TGN (Static Embeddings + Engineered Market Features).
    *   **Crucial**: Do NOT use any graph edges or message passing.
    *   Train on the same expanding window splits (2021-2024).
*   **Implementation Steps**:
    1.  Create `src/baselines/models_baseline.py`.
    2.  Define `BaselineMLP` with similar capacity to TGN Decoder (e.g., [Static+Market] -> 128 -> 64 -> 1).
    3.  Create `src/baselines/train_baseline.py` (strip out GNN components from `train_rolling.py`).
    4.  Compare AUC/F1 results. If TGN > Baseline, the graph structure adds value.

## 2. Priority: Technical Paper
**Goal**: Convert current findings into a publishable technical paper.

*   **Target Venue**: FinTech or AI applications conferences.
*   **Key Narrative**: "Temporal Graph Networks for Copy-Trading US Congress: Unlocking Relationships in Public Disclosures".
*   **To-Do**:
    *   Formalize the problem definition (Bipartite Temporal Graph).
    *   Include Ablation Study results (Politician Signal vs. Market Signal).
    *   Cite relevant literature on TGNs and Congressional Trading.

## 3. Graph Enhancements (Future Directions)
These features were identified as high-potential improvements in the presentation.

### A. Stock-Stock Relationships
*   **Idea**: Connect stocks based on Sector similarity (e.g., Tech-Tech) or Co-occurrence (traded by same politician).
*   **Expected Benefit**: Better market context propagation.

### B. Politician Social Graph
*   **Idea**: Connect politicians based on shared Committee Memberships.
*   **Hypothesis**: Committee peers might trade similarly due to shared information.
*   **Implementation**: Add `Politician --(Committee)--> Politician` edges.

### C. Dynamic Political Affiliation
*   **Idea**: Allow Party/State to be dynamic edge features rather than static node features.
*   **Benefit**: Captures party switching or changes in committee assignments over time.

## 4. Model Improvements

### A. Memory Decay Mechanism
*   **Problem**: Currently, TGN memory persists indefinitely. Old trades (e.g., from 2012) influence 2024 predictions equally if not updated.
*   **Solution**: Implement a time-based decay factor for the hidden state $h(t)$ to "forget" outdated information (e.g., > 2 years).

### B. Politician Success Scores
*   **Idea**: Create a dynamic feature tracking each politician's historical win-rate.
*   **Benefit**: Explicitly feeds "track record" to the model, reinforcing the politician signal.
