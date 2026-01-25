# Progress Update: Congressional Trading Prediction with TGNs
**Time Allocation:** ~10 Minutes
**Tone:** Direct, Objective, Impersonal.

---

## Section 1: Introduction & Motivation

### Slide 1: Title
> "This presentation provides a progress update on the Congressional Trading Prediction system using Temporal Graph Networks."

### Slide 2: Agenda
> "The agenda covers: Introduction, the 13-Experiment Evolution, the Current Architecture, the Evaluation Protocol, an Ablation Study, and Future Work."

### Slide 3: Project Overview
> "The objective is to predict if copying a Congressperson's stock trade will be significantly profitable—defined as a 1-month excess return versus SPY greater than 6%.
> This matters because Congress members have potential informational advantages through committee access and briefings, and their trades are publicly disclosed under the STOCK Act.
> The core challenge is that trades are temporal, politicians have interconnected behaviors, and traditional tabular ML ignores these relational dynamics."

### Slide 4: Why Temporal Graph Networks (TGN)?
> "The key insight is to model the market as a dynamic graph.
> Nodes are Politicians (~500) and Stocks (~2,000). Edges are Buy/Sell transactions at specific timestamps.
> Unlike Traditional ML where each trade is independent and has no memory, the TGN approach treats trades as connected, maintains memory per node, and uses dynamic features evolving over time.
> The result: the TGN can capture 'streaks' and learn 'who' trades 'what' effectively."

---

## Section 2: Model Evolution (13 Experiments)

### Slide 5: Experiment Journey: From Baseline to Final Model
> "13 systematic experiments were conducted to evolve the model:
> Exp 001-002: Baseline TGN + Dynamic Label Masking.
> Exp 003: Two-Phase Training (major breakthrough).
> Exp 004-005: Loss Function Experiments (Weighted BCE, Focal Loss).
> Exp 006: Validation of Best Configuration.
> Exp 007-010: Market Context Features.
> Exp 011: Deep TGN (2-Layer GNN).
> Exp 012: Deep Interaction Decoder.
> Exp 013: Scaling Experiments.
> Each component was tested independently to understand its contribution."

### Slide 6: Exp 001-003: Establishing the Foundation
> "Exp 001 added a `Masked_Label` feature: the historical trade outcome (Win/Lose) if resolved, else 0.5. Also added `Age`: how old each neighbor trade is, log-normalized. Result: High volatility (AUC: 0.42 to 0.81).
> Exp 002 introduced a 90/10 validation split with early stopping. Problem: Holding out 10% of recent data hurt model quality.
> Exp 003 was the major breakthrough: Two-Phase Training. Phase 1 finds the optimal epoch on 90% data. Phase 2 retrains from scratch on 100% data. Result: AUC jumped from 0.548 to 0.601 (+9.7%)."

### Slide 7: Exp 004-006: Loss Function Tuning
> "Exp 004 tested Weighted BCE + MeanAggregator. Result: Mixed. F1 improved (+0.029) but AUC dropped (-0.006). September collapsed (AUC 0.39).
> Exp 005 tested Focal Loss (alpha=0.25, gamma=2.0). Result: Failed. All metrics degraded. Focal Loss was too aggressive for this noisy dataset.
> Exp 006 reverted to Unweighted BCE + MeanAggregator. Confirmed: Two-Phase Training is the key driver. AUC: 0.596, Macro-F1: 0.569. Stable Baseline Established."

### Slide 8: Exp 007-010: Adding Market Context
> "Identified Gap: The model had no access to market conditions.
> Exp 007 added an LSTM encoder for 60-day price history (Stock + SPY). Result: Performance dropped. LSTM introduced noise.
> Exp 008 switched date basis from Trade Date to Filing Date. Result: Stability improved (Macro-F1 +5%).
> Exp 009-010 replaced LSTM with MLP on 14 engineered features: Return_1d, 5d, 10d, 20d, Volatility_20d, RSI_14, Vol_Ratio (x2 for Stock + SPY). Added BatchNorm1d + Dropout(0.2). Result: Macro-F1 recovered to 0.560. Key Improvement."

### Slide 9: Exp 011-012: Deepening the Architecture
> "Exp 011 introduced Deep TGN: Single TransformerConv became a 2-Layer Stack with ReLU + Dropout. Result: AUC 0.597 (New Best). More robust across months.
> Exp 012 introduced a Deep Interaction Decoder. Old: Sum(Src, Dst) -> ReLU -> Linear(1). New: Concat(Src, Dst) -> MLP(296->128->64->1). Rationale: Learn politician-stock interactions (e.g., 'Pelosi + Tech'). Result: AUC 0.600 – Broke the 0.60 barrier.
> Exp 013 scaled up (256 dims, 8 heads). Result: Overfitting. Performance regressed. Conclusion: Current capacity is sufficient."

---

## Section 3: Final Architecture

### Slide 10: Graph Construction
> "The data is represented as a Continuous Temporal Bipartite Graph.
> Nodes: Politicians (Source, ~500 unique) and Stocks (Destination, ~2,000 unique).
> Edges (Events): Represent a single BUY or SELL. Timestamp is Filing Date. Directed: Politician -> Stock.
> Dataset: Period 2012-2024 (~28,000 transactions). Source: Capitol Trades / House/Senate Stock Watcher. Label: Binary (1-Month Post-Filing Excess Return vs SPY > 6%)."

### Slide 11: Feature Engineering (Complete)
> "Dynamic Edge Features per Transaction: `Amount` (Log-normalized USD), `Is_Buy` (+1 Buy, -1 Sell), `Filing_Gap` (Days between Trade and Disclosure), `Masked_Label` (Historical outcome if resolved, else 0.5), `Age` (Log-normalized days since neighbor trade).
> Static Node Features: Party (8-dim) and State (8-dim) embeddings.
> Engineered Market Features (14 dims): Return_1d, 5d, 10d, 20d, Volatility_20d, RSI_14, Vol_Ratio. Computed for both target stock and SPY.
> Key Anti-Leakage: `Masked_Label` only reveals resolved trade outcomes."

### Slide 12: TGN Architecture (Final)
> "1. Price Encoder (MLP): Input: 14 engineered features -> Output: 32-dim embedding. Includes BatchNorm1d + Dropout(0.2).
> 2. Memory Module (GRU): Each node maintains hidden state h(t) = 100-dim. Message: [Amount, Is_Buy, Gap, Price_Emb] (35-dim). Aggregator: MeanAggregator.
> 3. Graph Embedding (2-Layer TransformerConv): 4 attention heads per layer, Dropout(0.1). Edge features: [Time_Enc, Msg, Masked_Label, Age].
> 4. Deep Interaction Decoder (MLP): Input: Concat(Z_src, Z_dst, S_src, S_dst, P_emb) = 296-dim. Architecture: 296 -> 128 -> 64 -> 1."

---

## Section 4: Evaluation Protocol

### Slide 13: Expanding Window Evaluation
> "Goal: Simulate real-world copy trading with no future information leakage.
> For each Test Month: Train on all data from 2012 to (Test Month - 1 month). Gap Phase: Most recent month—trades exist, but labels are unknown. Forward-only mode (update memory, no backprop). Test: Predict trades in the target month.
> Two-Phase Training (per month): Phase 1 trains on 90% of data, finds best epoch via early stopping (patience=5). Phase 2 retrains a fresh model on 100% data for best_epoch epochs.
> Metrics: ROC-AUC, PR-AUC, F1-Score, Macro-F1."

---

## Section 5: Ablation Study

### Slide 14: Ablation Study: Which Signal Matters?
> "Objective: Isolate the contribution of each feature group over 6 years (2019-2024).
> Configurations Tested:
> 1. Politician Signal Only (`pol_only`): Graph + Memory + Static Embeddings (Party, State). Zero out all 14 market features.
> 2. Market Signal Only (`mkt_only`): 14 Engineered Market Features. Zero out Static Embeddings.
> 3. Full Model (`full`): All features active.
> Scale: 72 months x 3 modes = 216 model training runs."

### Slide 15: Ablation Results: Yearly F1 Score
> "Mean F1 Scores: Pol Only: 0.452. Mkt Only: 0.471 (Highest). Full: 0.443.
> Observation: `mkt_only` achieves highest F1 (0.471)—market context helps classification calibration."

### Slide 16: Ablation Results: Yearly ROC-AUC
> "Mean AUC Scores: Pol Only: 0.560 (Highest). Mkt Only: 0.553. Full: 0.551.
> Observation: `pol_only` achieves highest AUC (0.560)—politician identity is a strong ranking signal."

### Slide 17: Ablation Visualization (Trend)
> "The chart shows Monthly F1 Trend (3-Month Rolling Average). All models improve from 2019 to 2024."

### Slide 18: Ablation Visualization (Yearly F1)
> "The bar chart shows Yearly Average F1 Score by Signal Source (2019–2024). Market Only generally leads."

---

## Section 6: Discussion & Conclusions

### Slide 19: Key Findings
> "1. Two-Phase Training is Critical: Exp 003 showed +9.7% AUC improvement from this technique alone. Using all recent data for backprop is essential.
> 2. Engineered Features > Raw Sequences: LSTM on raw OHLCV failed (Exp 007). MLP on engineered features (RSI, Returns, Vol) succeeded (Exp 009-010).
> 3. Politician Signal is Robust for Ranking: Ablation shows highest AUC (0.560) using only graph + static embeddings.
> 4. Full Model Shows No Clear Synergy: Possible causes: Feature interference, overfitting, or suboptimal fusion."

### Slide 20: Addressed Challenges & Iterations
> "Memory underutilization: Fixed. Switched to MeanAggregator.
> Missing market context: Fixed. Added 14 engineered features.
> No validation set: Fixed. Two-phase training.
> Class imbalance: Partial. Tested Focal Loss (failed).
> Extreme volatility: Partial. Improved but still present.
> Stock-stock relationships: Open. Future work."

### Slide 21: Limitations
> "Label Resolution Delay: 1-month labels are noisy (short-term volatility).
> Sample Size Variance: Monthly counts range from 100 to 600. Small samples cause high AUC variance.
> Feature Fusion: Concatenation may not be optimal. Attention-based fusion is unexplored.
> Graph Structure: No stock-to-stock relationships (sector correlations not captured).
> Survivorship Bias: Only trades from active members are included."

---

## Section 7: Future Work

### Slide 22: Future Directions: Enhancing the Graph
> "1. Stock-Stock Relationships: Add sector similarity edges. Add co-occurrence edges (stocks traded together).
> 2. Politician Social Graph: Add Politician-Politician edges based on shared committee memberships. Tests hypothesis: 'Do committee peers trade similarly?'
> 3. Dynamic Political Affiliation: Currently, Party is static (first observed). Future: Attach Party as an edge feature to capture party switching or committee changes over time."

### Slide 23: Future Directions: Model & Strategy
> "1. Advanced Fusion Mechanisms: Cross-Attention between Static/Market branches. Gated Fusion Networks.
> 2. Politician Success Scores (Dynamic Feature): Historical win-rate per politician (updated monthly). Leverage BioGuideID to track long-term performance.
> 3. Copy-Trading Strategy Backtesting: Simulate portfolio returns based on model predictions. Metrics: Sharpe Ratio, Max Drawdown, Calmar Ratio.
> 4. Memory Decay Mechanism: Implement time-based decay for old memory states h(t). Reduce influence of outdated trading patterns (e.g., > 2 years)."

### Slide 24: Thank You
> "This concludes the progress update. The baseline is validated (AUC > 0.60) with a clear technical path forward. Questions are welcome."
