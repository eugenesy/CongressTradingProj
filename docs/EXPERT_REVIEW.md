# Expert Review: TGN-Based Congressional Copy Trading System

**Reviewer Profile**: Senior Researcher, 10+ years experience in Graph Neural Networks, Temporal Networks, and Quantitative Finance.

---

## Executive Summary

This system attempts to predict profitable copy-trading opportunities from US Congressional stock transactions using a Temporal Graph Network (TGN). While the approach demonstrates **solid engineering** and **correct temporal handling**, there are several **critical concerns** that could limit its real-world applicability and inflate reported performance.

**Overall Assessment**: 🟡 **Promising but Requires Significant Revisions**

---

## I. Strengths

### 1. Correct Temporal Handling ✅
- The "Predict-then-Update" cycle is correctly implemented.
- Resolution time masking (`is_resolved = hist_resolution < batch_max_t`) prevents future label leakage.
- Monthly retraining ("Level 2") is a rigorous evaluation protocol.

### 2. Reasonable Feature Engineering ✅
- Static features (Party, State) provide meaningful signal.
- Dynamic features (Filing Gap, Amount, Direction) capture trade characteristics.
- The new "historical label" feature is a clever way to encode track records.

### 3. Appropriate Architecture ✅
- TGN is well-suited for this problem (sequential events, node memory).
- TransformerConv with edge features allows flexible neighbor weighting.

---

## II. Critical Concerns

### 🔴 1. Sample Size and Statistical Significance

**Problem**: The monthly sample sizes are **dangerously small**.

| Month | Count | Concern |
|-------|-------|---------|
| 2024-03 | 108 | Extremely low |
| 2022-05 | 123 | Extremely low |
| 2024-02 | 125 | Extremely low |

With ~100-200 samples per month, the **variance of AUC estimates is enormous**. A swing from 0.40 to 0.75 could easily be noise.

**Recommendation**:
- Report **confidence intervals** (bootstrap 95% CI).
- Aggregate results across quarters or years before drawing conclusions.
- Conduct **statistical significance tests** (paired t-test across months).

---

### 🔴 2. Class Imbalance Not Addressed

**Problem**: The dataset likely has **significant class imbalance** (more wins or more losses depending on market conditions).

**Evidence**: Some months show Accuracy < 50% with AUC > 0.5. This happens when the model predicts the majority class but struggles with the minority.

**Recommendation**:
- Report **class distribution** per month.
- Use **balanced BCE loss** or focal loss.
- Report **Precision-Recall AUC (PR-AUC)** instead of or in addition to ROC-AUC.
- Consider **stratified evaluation** by class.

---

### 🔴 3. Extreme Volatility in Results

**Problem**: Monthly AUC ranges from **0.38 to 0.75**. This is a **37-point swing**.

```
2023-09: AUC 0.38 (Disaster)
2023-10: AUC 0.75 (Excellent)
2023-11: AUC 0.39 (Disaster)
2023-12: AUC 0.73 (Excellent)
```

**Interpretation**: The model is **not stable**. This could indicate:
1. **Overfitting to specific regime patterns** that don't generalize.
2. **Market regime shifts** that the model cannot adapt to.
3. **Random noise** due to small sample sizes.

**Recommendation**:
- Implement **regime detection** (e.g., VIX-based, market return-based).
- Train **separate models** per regime or use regime as a feature.
- Investigate **what changed** between 2023-09 and 2023-10.

---

### 🔴 4. Memory Module Underutilization

**Problem**: You are using `IdentityMessage` and `LastAggregator`.

```python
message_module=IdentityMessage(raw_msg_dim, memory_dim, time_dim),
aggregator_module=LastAggregator(),
```

**Concern**: 
- `IdentityMessage` simply concatenates the raw message with memory. It does not learn a transformation.
- `LastAggregator` takes only the **most recent** message per node. If a politician has 5 trades in a day, only the last one affects memory.

**Recommendation**:
- Use `MLPMessage` or a custom learned message function.
- Use `MeanAggregator` or attention-based aggregation to incorporate multiple concurrent events.

---

### 🟡 5. Graph Structure Limitations

**Problem**: The graph is **bipartite** (Politician → Stock). There is no **stock-to-stock** relationship.

**Missed Signal**:
- Politicians often trade **related stocks** (e.g., all tech stocks).
- Sector/industry correlations are not captured.
- If Pelosi buys NVDA and simultaneously sells AMD, the model doesn't see the **relationship** between NVDA and AMD.

**Recommendation**:
- Add **stock co-occurrence edges** (stocks traded together by any politician).
- Add **sector similarity edges** using sector/industry metadata.
- Consider a **hypergraph** where each politician's "trade bundle" is a hyperedge.

---

### 🟡 6. Missing Market Context

**Problem**: The model has **no access to market conditions**.

**Missed Signal**:
- Is the market in a bull or bear regime?
- What is the VIX (volatility index)?
- What is the recent momentum of the target stock?

**Evidence**: The model fails catastrophically in certain months (2023-09, 2024-01), which correlate with **market stress periods**.

**Recommendation**:
- Add **SPY return** (trailing 5-day, 20-day) as a global feature.
- Add **VIX level** as a feature.
- Add **stock momentum** (trailing 5-day return) as an edge feature.

---

### 🟡 7. Evaluation Protocol Issues

**Problem 1**: No validation set for hyperparameter tuning.
- The model is trained and tested without a separate validation phase.
- This means hyperparameters (learning rate, embedding dim, etc.) may be **overfit to test data** through implicit tuning.

**Problem 2**: No comparison to baselines.
- What is the performance of a simple **majority class predictor**?
- What is the performance of **logistic regression** on the same features?
- Without baselines, it's impossible to know if TGN adds value.

**Recommendation**:
- Implement **nested cross-validation** or a held-out validation period.
- Compare against:
  1. Majority class baseline
  2. Random baseline (AUC ≈ 0.50)
  3. Feature-based logistic regression
  4. XGBoost on engineered features

---

### 🟡 8. Potential Information Leakage (Subtle)

**Concern**: The `Filing Gap` feature uses `Filed_Date - Traded_Date`.

**Question**: When is `Filed_Date` known?
- If `Filed_Date` is known **at filing time** (not trade time), then using it means the model can only make predictions **after the filing** is public.
- But the label (`Label_1M`) is computed from **Trade Date + 30 days**.

**Scenario**:
- Trade: Jan 1
- Filed: Jan 20
- Resolution: Feb 1
- If we predict on Jan 20, we only have 10 days until resolution.

**Recommendation**:
- Clarify the **prediction timing** (at Trade, at Filing, or at some other point).
- If prediction is at Filing time, the **Label_1M** should be recomputed from Filing date.

---

## III. Recommended Improvements (Priority Order)

| Priority | Improvement | Impact |
|----------|-------------|--------|
| 🔴 High | Add confidence intervals and statistical tests | Credibility |
| 🔴 High | Address class imbalance (balanced loss, PR-AUC) | Correctness |
| 🔴 High | Add baseline comparisons | Validity |
| 🟡 Medium | Add market context features (SPY, VIX) | Performance |
| 🟡 Medium | Upgrade message/aggregator modules | Performance |
| 🟡 Medium | Add stock-stock relationships | Performance |
| 🟢 Low | Regime detection and conditioning | Stability |

---

## IV. Questions for the Authors

1. What is the **average AUC** across all months, with **standard deviation**?
2. What is the **class distribution** (% wins) per year?
3. Have you compared against a **buy-and-hold SPY** strategy?
4. What is the **actual Sharpe ratio** if trades are executed based on model predictions?
5. Why was `LastAggregator` chosen over `MeanAggregator`?

---

## V. Conclusion

This is a **technically competent** implementation of TGN for a novel financial application. However, the **evaluation is incomplete** and the **results are too volatile** to draw strong conclusions. Before claiming that the model "works," the authors should:

1. Add proper statistical analysis.
2. Compare against simple baselines.
3. Investigate the extreme month-to-month variance.
4. Consider adding market context features.

The core idea (using Congressional trading as a signal) is interesting and potentially viable(enough), but the current evidence is not sufficient to support deployment.

---

*— Expert Reviewer, December 2024*
