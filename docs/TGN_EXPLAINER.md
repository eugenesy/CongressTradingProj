# How TGN Works for Copy Trading

## The Problem We're Solving

**Goal**: Predict if copying a politician's stock trade will be profitable (1-month return > 0%).

**Challenges**:
1. Trades happen over time (2015-2024). Old patterns may not apply today.
2. Politicians have trading "streaks" (hot/cold periods).
3. Some stocks are consistently targeted by multiple politicians.

---

## Why TGN? (vs Traditional ML)

| Traditional ML | TGN |
|----------------|-----|
| Each trade is independent | Trades are connected (same politician, same stock) |
| No memory of past trades | Explicit memory per politician/stock |
| Static features only | Dynamic features that evolve with time |

---

## The TGN Architecture (Our Implementation)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TGN PIPELINE                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. INPUT: A new trade arrives                                      │
│     └─ [Politician A] ──buys──▶ [Stock X] at time T                │
│                                                                     │
│  2. MEMORY LOOKUP                                                   │
│     └─ Retrieve hidden states: h_A (politician), h_X (stock)       │
│        These compress ALL past trades for A and X.                  │
│                                                                     │
│  3. NEIGHBOR LOOKUP                                                 │
│     └─ Find recent trades involving A or X                          │
│        "What else has A traded? Who else bought X?"                 │
│                                                                     │
│  4. GRAPH ATTENTION (TransformerConv)                               │
│     └─ Aggregate neighbor information with attention weights        │
│        Recent/important neighbors get higher weight.                │
│                                                                     │
│  5. STATIC FEATURES                                                 │
│     └─ Add politician metadata: Party (Dem/Rep), State (CA/TX)      │
│                                                                     │
│  6. PREDICTION                                                      │
│     └─ MLP predicts: P(Win) = sigmoid(W · [h_A, h_X, static])       │
│                                                                     │
│  7. MEMORY UPDATE (After Prediction)                                │
│     └─ Update h_A and h_X with THIS trade's info.                   │
│        "Now they know about this trade for future predictions."     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Training Strategy: "Level 2" Rolling Window

We use **Strict Monthly Retraining** to simulate real-world copy trading.

### The Loop (For each month Jan 2021 → Dec 2024):

```
┌─────────────────────────────────────────────────────────────────────┐
│  PREDICTING: March 2023                                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  PHASE 1: TRAIN (2015 → Jan 2023)                                   │
│  ├─ Use ALL historical trades with KNOWN labels                     │
│  ├─ Train model weights (5 epochs)                                  │
│  └─ Build memory for all politicians/stocks                         │
│                                                                     │
│  PHASE 2: GAP (Feb 2023)                                            │
│  ├─ Trades exist, but labels are UNKNOWN (not resolved yet)         │
│  ├─ Forward pass ONLY (no training)                                 │
│  └─ Update memory so model has latest context                       │
│                                                                     │
│  PHASE 3: TEST (March 2023)                                         │
│  ├─ Predict each trade                                              │
│  ├─ Compare to actual labels                                        │
│  └─ Report AUC, Accuracy, F1                                        │
│                                                                     │
│  PHASE 4: MOVE TO NEXT MONTH                                        │
│  └─ Retrain from scratch for April 2023                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Why Retrain Every Month?
- Markets change (regime shifts).
- A model trained in 2020 won't understand 2023 patterns.
- Monthly retraining lets the model adapt.

---

## How Training Works (Incremental Graph Building)

The graph is NOT built all at once. It grows **transaction by transaction**:

```
EPOCH 1 (of 5):
├─ RESET: Clear Memory & Graph
│
├─ Batch 1: Jan 2015 (200 trades)
│   ├─ Graph: EMPTY → Predict (random)
│   ├─ Backprop → Update Weights
│   ├─ Update Memory for involved nodes
│   └─ Add 200 edges to graph
│
├─ Batch 2: Feb 2015 (200 trades)
│   ├─ Graph: 200 edges → Can see Jan history!
│   ├─ Predict using neighbor features
│   ├─ Backprop → Update Weights
│   └─ Graph now has 400 edges
│
├─ ... (all batches) ...
│
└─ Batch N: Jan 2023
    └─ Graph: ~20,000 edges of full history
```

**Key Insight**: When predicting Feb 2020 trades:
- Model sees all trades from 2015-Jan 2020 as neighbors
- Model knows labels for trades that resolved (2015-Dec 2019)
- This is exactly what a real trader would know!

---

## Validation vs Testing

| Phase | Data | Purpose |
|-------|------|---------|
| **Train** | 2015 → (Month - 2) | Learn patterns, optimize weights |
| **Gap** | (Month - 1) | Update memory without labels (context) |
| **Test** | Target Month | Final evaluation, metrics reported |

> **Note**: We do NOT use a separate validation set in the current rolling setup. Each month is evaluated independently. If we wanted to tune hyperparameters, we would add a validation phase.

---

## What Makes This Work

1. **Memory**: Politicians/stocks accumulate a "reputation" over time.
2. **Attention**: Recent trades matter more than old ones.
3. **Static Features**: Party affiliation provides signal (Dems vs Reps invest differently).
4. **Dynamic Labels**: Historical resolved trades show "track record" (NEW FEATURE).
5. **No Leakage**: Labels only visible after resolution time.

---

## Metrics We Report

| Metric | What It Measures |
|--------|------------------|
| **AUC** | Ranking quality (can the model separate winners from losers?) |
| **Accuracy** | Overall correctness |
| **F1 Score** | Balance of precision and recall (handles class imbalance) |
