# lapulga-llm 🐐

*(La Pulga is intentionally named as a tribute to Lionel Messi – reflecting our goal to build an architecture that is extremely small, highly agile, and exceptionally high-performing, just like him).*

## The Parameter Golf Challenge

Runpod is OpenAI's infrastructure partner on Parameter Golf, the first challenge in the OpenAI Model Craft Challenge series.

Train the best language model that fits in a **16MB artifact** in under **10 minutes** on **8×H100s**. Submissions run through a public GitHub-based leaderboard. It's open to researchers, engineers, and builders of all kinds.

---

## 🧠 Architecture Setup (v6.0 - The "10L int6 Gated Leaky-U-Net" Record Candidate)

To respect the **16MB boundary**, *La Pulga LLM* now targets a **shared-weights ALBERT-style architecture with 15,699,158 stored parameters** and int6+zlib export. Effective depth is preserved through 4 physical layers repeated 3 times (12 effective transformer steps).

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Vocab Size (V)** | `1,024` | Official SentencePiece sp1024 vocabulary for challenge compatibility. |
| **Dim (D)** | `768` | Core hidden dimension aligned with H100 optimization targets. |
| **Physical Layers** | `10` | Unique parameterized blocks stored on disk. |
| **Repeat Count** | `1` | No sharing. |
| **Attention Heads** | `12` | 64 dimensions per head. |
| **KV Heads (GQA)** | `4` | Grouped-Query Attention to share K&V weights. |
| **MLP Hidden** | `4096` | LeakyReLU² feed-forward width tuned to stay below 16,000,000-byte artifact limit. |

## ⚡ Tech Stack & Optimizations

*   **Apple Silicon (M4):** Local prototyping is completely driven by [Apple's MLX Framework](https://github.com/ml-explore/mlx). We write training loops in Python taking advantage of the unified memory for ultra-fast local iteration batches.
*   **Runpod (8xH100):** Cloud instances will leverage **PyTorch**, **FlashAttention-2** and potentially **Unsloth** wrappers to squeeze every microsecond out of the 10-minute training wall.
*   **Weight Tying:** The model explicitly ties the Unembedding matrix weights back to the Input Embeddings, recovering nearly `~2M` parameters instantly. We also use **Factored Embeddings** (254-dim mapped to 768-dim) to save an additional ~0.5MB, allowing us to dramatically expand the MLP width.
*   **Checkpointing + Export:** We save `.safetensors` and emit official int6+zlib artifacts for strict 16,000,000-byte validation.

## 📂 Project Structure

*   **/docs**: Real-time tracking of architectural mathematics, logs (`journal.md`), terminology (`glossary.md`), and historical run data (`benchmarks.md`).
*   **`train_mlx.py`**: Our primary script for testing parameter golf locally on macOS constraints.
