# lapulga-llm 🐐

*(La Pulga is intentionally named as a tribute to Lionel Messi – reflecting our goal to build an architecture that is extremely small, highly agile, and exceptionally high-performing, just like him).*

## The Parameter Golf Challenge

Runpod is OpenAI's infrastructure partner on Parameter Golf, the first challenge in the OpenAI Model Craft Challenge series.

Train the best language model that fits in a **16MB artifact** in under **10 minutes** on **8×H100s**. Submissions run through a public GitHub-based leaderboard. It's open to researchers, engineers, and builders of all kinds.

---

## 🧠 Architecture Setup (v0.1 - Baseline)

To respect the **16MB boundary**, *La Pulga LLM* relies on an extremely compact parameter budget. The current baseline design targets **~7.8M parameters** (stored in `FP16` precision), totaling exactly **~15.6 MB** on disk.

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Vocab Size (V)** | `8,192` | Extremely restricted to avoid embedding matrix bloat. |
| **Dim (D)** | `256` | Core hidden dimension. |
| **Layers (L)** | `6` | Transformer depth (Attention + MLP). |
| **Attention Heads** | `8` | 32 dimensions per head. |
| **KV Heads (GQA)** | `2` | Grouped-Query Attention to share K&V weights, saving massive space. |
| **MLP Hidden** | `1024` | Feed-Forward expansion space (SwiGLU). |

## ⚡ Tech Stack & Optimizations

*   **Apple Silicon (M4):** Local prototyping is completely driven by [Apple's MLX Framework](https://github.com/ml-explore/mlx). We write training loops in Python taking advantage of the unified memory for ultra-fast local iteration batches.
*   **Runpod (8xH100):** Cloud instances will leverage **PyTorch**, **FlashAttention-2** and potentially **Unsloth** wrappers to squeeze every microsecond out of the 10-minute training wall.
*   **Weight Tying:** The model explicitly ties the Unembedding matrix weights back to the Input Embeddings, recovering nearly `~2M` parameters instantly.
*   **Checkpointing:** We dump raw bytes using `.safetensors` on every epoch validation to ensure no hidden overhead breaches max capacity.

## 📂 Project Structure

*   **/docs**: Real-time tracking of architectural mathematics, logs (`journal.md`), terminology (`glossary.md`), and historical run data (`benchmarks.md`).
*   **`train_mlx.py`**: Our primary script for testing parameter golf locally on macOS constraints.
