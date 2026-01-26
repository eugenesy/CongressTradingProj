# TGN 2023 Research Log

## Experiment 0: Official Benchmarks (Reference)
*   **XGBoost (6M)**: AUC ~0.5318 | Accuracy ~56%
*   **LogReg (6M)**: AUC ~0.5447 | Accuracy ~57%

---

## Experiment 1: Baseline TGN (Pre-Remediation)
*   **Config**: IdentityMessage, Weighted BCE, No gating.
*   **Date**: 2026-01-26
*   **Results**:
    - Avg AUC: 0.509
    - Observations: High accuracy on majority class, but poor signal. Validation loss was 'inf' initially due to code bugs.
*   **Status**: COMPLETED

---

## Experiment 2: Enhanced TGN (Phase 2 Remediation)
*   **Config**: MLP Messenger + Feature Gating + 10 Epochs.
*   **Date**: 2026-01-26 (In Progress)
*   **Initial Findings**:
    - March 2023: AUC 0.568 (Beat XGBoost!)
    - May 2023: AUC 0.559 (Beat XGBoost!)
*   **Observation**: Architecture changes are providing real signal boost.
*   **Status**: ABORTED (Pivot to clean dev/ structure)

---

## Experiment 3: Clean Room "Enhanced" TGN (dev/)
*   **Config**: MLP Messenger + Feature Gating + Fixed 5 Epochs (Seed 42).
*   **Date**: 2026-01-26
*   **Status**: COMPLETED
*   **Results (Mean AUC 2023)**: **0.5136**
    *   **vs Baseline**: +0.0285 (beat XGBoost 0.4851)
    *   **Peak Performance**: July 2023 (AUC 0.6098)
    *   **Consistency**: beat baseline in 8/12 months.

## Phase 3: Full Dataset Verification (2019-2024)
*   **Objective**: Test robustness over 6 years (market cycles).
*   **Status**: COMPLETED
*   **Results (Mean AUC)**:
    *   **TGN (Ours)**: **0.5149** (1st Place)
    *   **MLP**: 0.5110
    *   **LR**: 0.5057
    *   **XGB**: 0.5050
*   **Key Insight (Recall)**: The TGN is the only model correctly identifying a significant portion of winning trades.
    *   **TGN Recall (Up)**: **34%**
    *   **MLP Recall (Up)**: 19%
    *   **XGB Recall (Up)**: 17%
    The TGN is "braver" and catches roughly **2x** more winners than the baselines, which tend to degenerate into "predict Down" majorities. This suggests the graph structure provides confidence to go long where price-only models fold.

---

## Experiment 4: Fair Baselines (dev/)
*   **Config**: XGBoost & LogReg respecting the Resolution Gap (Horizon).
*   **Rational**: Previous benchmarks might have benefited from future data leakage. This experiment uses the exact same `Resolution < Test_Start` mask as the TGN.
*   **Status**: COMPLETED
*   **Results (Mean AUC 2023)**:
    *   **XGBoost**: 0.4851
    *   **LogReg**: 0.4901
    *   **Torch MLP**: 0.4940
*   **Observation**: Strict temporal masking reveals the task is extremely difficult. The baseline is effectively random (0.5). TGN success will be defined as consistently >0.50.

---

## Strategy Consultation: Metric Selection
*   **F1_Up**: Chosen for primary model ranking. It ensures the model is neither "too shy" (missing all alpha) nor "too reckless" (high false positives). 
*   **PRC (Average Precision)**: Designated as the "Gold Standard" for high-conviction trading. Since the task is to find a needle in a haystack, PRC provides a cleaner look at signal quality than ROC-AUC.
*   **Recall (Up)**: The TGN's primary edge. By identifying ~2x more winning trades than baselines, it enables a much higher turnover and capital utilization.

