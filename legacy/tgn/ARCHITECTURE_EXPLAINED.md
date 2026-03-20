# Node-Centric TGN: System Architecture & Workflow

## 1. The Core Philosophy
Traditional stock prediction treats every trade as an isolated row in a table (XGBoost).
**Our Approach (TGN)** treats the market as a **Living Social Network** of politicians and companies.
*   **The Idea**: Trades are not isolated. If Pelosi buys Apple today, it changes her "State". If she buys Google tomorrow, she brings her "Apple Experience" (and track record) with her.
*   **The Network Effect**: We reconstruct the network dynamically by linking politicians to the companies they trade, and "remembering" these links over time.

---

## 2. Feature Engineering (The Inputs)
Every Trade Event $(u, v, t)$ is enriched with **4 Critical Signals** before it touches the Neural Network.

### A. The "Rolling Win Rate" (The Amnesia Fix)
*   **Problem**: TGN Memory usually forgets long-term history.
*   **Solution**: We explicitly calculate the "Track Record" for every single row.
*   **Logic**:
    ```python
    Win_Rate(t) = (Wins prior to t) / (Total Resolved Trades prior to t)
    ```
    *   *Strict Rules*: A trade is only counted if its outcome (Resolution Date) is known *before* the current Filing Date. **Zero Look-ahead Bias.**

### B. The "Filing Gap" (Psychology)
*   **Logic**: `Gap = Filing_Date - Trade_Date`
*   **Signal**:
    *   **Short Gap (1-3 days)**: "Routine, confident."
    *   **Long Gap (45+ days)**: "Hiding, hesitation, or forgetfulness."
*   **Implementation**: Fed as `log(Gap_Days)` into the Message Vector.

### C. Static Context (Node Features)
*   Party (Dem/Rep)
*   Chamber (House/Senate)
*   State (CA, TX, etc.)
*   *Note*: These are embedded and fused with the dynamic signal.

---

## 3. The Architecture (Node-Centric TGN)
We use a customized **Temporal Graph Network (TGN)** with specific modifications for low-frequency financial data.

### The Components
1.  **The Memory (The Diary)**:
    *   Every Node (Politician & Company) has a vector $S_u$.
    *   This vector persists strictly. It is NOT reset between batches. It bridges the months of silence between trades.
2.  **The Message (The News)**:
    *   Event: "Pelosi buys NVIDIA".
    *   Message: `[Amount, Buy/Sell, Gap_Days, Win_Rate]`.
3.  **The Updater (GRU)**:
    *   Update: $S_u(new) = GRU(S_u(old), Message)$.
4.  **The Neighbor Loader (The Ghost Network)**:
    *   When predicting for Pelosi, we look at her last **10 connections**.
    *   This aggregates her recent history into a "Context Embedding" $Z_u$.

### The Predictor (MLP)
We feed 3 vectors into a Multi-Layer Perceptron to output a probability:
$$ P(Win) = MLP( Z_{Pelosi} \oplus S_{Static} \oplus Z_{Company} ) $$
*(Note: We removed the implicit ID bias to make the model "Inductive" — capable of handling new senators)*.

---

## 4. The "Rolling Evaluation" Protocol (Training Loop)
To simulate real-world trading perfectly, we use a **Monthly Rolling Window**.

**Example: Simulating February 2023**
1.  **RESET**: We wipe the model's memory completely.
2.  **REPLAY (Train)**: We stream the *entire* history from 2015 up to Jan 31, 2023.
    *   The model "re-lives" 8 years of data to calibrate Pelosi's memory to the exact state of Jan 31.
3.  **PREDICT (Test)**: We step through February day-by-day.
    *   **Trade 1**: Predict -> Compare to Label.
    *   **update**: **Immediately update Memory** with Trade 1 (Transductive).
    *   **Trade 2**: Predict (using updated memory).
    *   *This simulates a live trading bot exactly.*

---

## 5. Current Status (As of Run 007)
*   **Performance**: ~0.53 AUC (Beats Baseline 0.51).
*   **Behavior**: Distinct **Momentum Detector**.
    *   High Scores (0.60+) in Momentum Markets (Feb, Jun).
    *   Low Scores (<0.50) in Reversion Markets (Jan, May).
### D. Market & Stock Context (Price Encoder) - [NEW]
*   **Problem**: TGN doesn't know if the market is crashing or mooning.
*   **Solution**: We attach a 14-dimensional "Price Context" vector to every trade.
*   **Logic**:
    *   **Stock Signals**: Return (1d/5d/10d/20d), Volatility (20d), RSI (14d), Volume Ratio.
    *   **Market (SPY) Signals**: Identical set of 7 features for the S&P 500.
*   **Result**: The model now has a "Market Compass" to differentiate between a bull run and a bear trap.

---

## 4. The Architecture (Node-Centric TGN)
... (components remain as described) ...

### The Predictor (Deep Fusion MLP)
We feed 4 vectors into a Multi-Layer Perceptron:
$$ P(Win) = MLP( Z_{User} \oplus Z_{Company} \oplus Context_{Price} \oplus Features_{Static} ) $$

---

## 5. The "Rolling Evaluation" Protocol (Training Loop)
... (protocol remains as described) ...

---

## 6. Current Status (Complete Pipeline)
*   **Data Quality**: ~34k transactions, 0 disclosure lag NaNs.
*   **Features**:
    *   ✅ Rolling Win Rate
    *   ✅ Filing Gap (Log-scaled)
    *   ✅ 14-dim Multi-Horizon Price Context
*   **Performance**: The system is now baseline-ready for multi-class classification and dynamic thresholding.
*   **Key Insight**: By anchoring to **Filing Date**, we have eliminated "Look-ahead Bias" entirely.
