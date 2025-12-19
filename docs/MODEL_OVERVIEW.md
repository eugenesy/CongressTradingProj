# Temporal Graph Network (TGN) for Politician Copy Trading

## 1. Data Representation
The market is modeled as a **Continuous Temporal Bipartite Graph**.

*   **Nodes**:
    *   **Source**: Politicians (Dynamic Agents).
    *   **Destination**: Companies (Stocks).
*   **Edges (Events)**:
    *   Represent a single Transaction (`Buy` or `Sell`) at a specific timestamp $t$.
    *   **Directed**: $Politician \rightarrow Company$ (Action flow).

### Features
We utilize a rich set of dynamic and static features to capture signal:
1.  **Dynamic Edge Features** (Time-Variant):
    *   `Amount`: Log-normalized USD transaction size.
    *   `Is_Buy`: Directional indicator (+1 for Buy, -1 for Sell).
    *   `Filing_Gap`: Log-normalized days between `Trade_Date` and `Filing_Date` (captures reporting latency/urgency).
2.  **Static Node Features** (Time-Invariant):
    *   `Party Embedding`: Learnable vector for Political Party (Dem/Rep).
    *   `State Embedding`: Learnable vector for Home State (CA, TX, NY, etc.).

---

## 2. Architecture
The model is a **Temporal Graph Network (TGN)** based on the Twitter Research architecture (Rossi et al.), customized for financial time-series.

### A. Memory Module (The "Brain")
*   **State**: Each node (Politician and Company) maintains a hidden state vector $h(t)$.
*   **Function**: $h(t)$ compresses the entire history of the node into a fixed-size vector.
*   **Update Mechanism**: When a trade occurs, the memory is updated using a **GRU (Gated Recurrent Unit)**:
    $$ h(t) = GRU(\text{Current Event}, h(t-1)) $$
*   **Significance**: Allows the model to remember long-term behaviors (e.g., "Pelosi has been accumulating Tech stocks for 6 months").

### B. Graph Embedding Module (The "Context")
*   **Layer**: `TransformerConv` (Graph Transformer).
*   **Mechanism**: To predict an edge, the model aggregates information from the node's **Spatial Neighbors** (recent trading partners).
*   **Time Encoding**: Relative time differences ($t_{now} - t_{neighbor}$) are encoded using learnable Sinusoidal embeddings (like Transformers) to weight recent events higher than old ones.

### C. Link Predictor (The "Decision")
*   **Input**: Concatenation of:
    *   Dynamic Politician Embedding $Z_{src}$
    *   Dynamic Company Embedding $Z_{dst}$
    *   Static Party/State Embeddings $S_{src}, S_{dst}$
*   **MLP**: A multi-layer perceptron outputs a probability $P(Win)$.
*   **Task**: Binary Classification (Will this trade return > 0% in 1 Month?).

---

## 3. Methodology

### "Predict-then-Update" Cycle
To prevent **Data Leakage** (Cheating), we strictly follow this cycle for every batch of trades:
1.  **Retrieve State**: Get current Memory $h$ for involved nodes.
2.  **Predict**: Estimate probability of success for the current batch of trades.
3.  **Compute Loss**: Compare prediction to ground truth (`Label_1M`).
4.  **Update Memory**: *Only after prediction*, update the Memory $h$ with the new trade information.
5.  **Grow Graph**: Add the new edges to the graph structure for future neighbors.

This ensures the model never uses the existence of a trade to predict the trade itself.

---

## 4. Evaluation Method: "Level 2" Rolling Window

We employ a **Strict Realistic Backtesting** strategy to simulate a real-world hedge fund.

### Strategy
*   **Retraining Frequency**: Monthly.
*   **Window**: Rolling.

### The Algorithm (For every Test Month $M$):
1.  **Define Training Set**: All history from `2015-01-01` up to `Start of Month M - 1 Month`.
2.  **Train**: Initialize a **Fresh Model** and train weights from scratch on the Training Set.
3.  **Process Gap (Reality Check)**:
    *   **The Gap**: The period `[M-1 Month, M]` (The most recent 30 days).
    *   **Constraint**: We know the *trades* happened (Edges), but we *don't know their labels yet* (because the 1-month return hasn't materialized).
    *   **Action**: Feed Gap trades into the model in **Forward-Only Mode** (Update Memory/Graph, but NO Backprop/Training).
    *   **Result**: The model's Memory is up-to-the-minute, but it hasn't cheated by seeing future labels.
4.  **Test**: Predict trades in Month $M$.
5.  **Record**: Save AUC/Accuracy.

This method is computationally expensive (requires retraining hundreds of times) but provides the highest possible confidence that the results are robust and tradeable.
