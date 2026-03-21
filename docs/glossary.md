# lapulga-llm: Glossary

- **Parameter (weight/knob):** The learned variables (weights and biases) of the neural network optimized during training. In the context of the 16MB constraint, managing the total parameter count or compressing each parameter is the primary challenge to fitting the model into the strict storage budget.
- **Quantization (bit-reduction for space):** The process of reducing the numerical precision of the model's weights (e.g., from 16-bit floats to 4-bit integers). This is a critical strategy for maximizing the model's capacity under the 16MB limit, allowing the architecture to hold up to ~33.5M parameters (at 4-bit) instead of just ~8.3M (at 16-bit).
- **Perplexity (error metric for quality):** The standard evaluation metric that measures how well the language model predicts the next token. Given the strict 16MB boundary and the 10-minute training window, the core objective is to tune the architecture and hyperparameters to achieve the lowest possible perplexity without violating these constraints.
- **Embedding (word-to-vector mapping):** The lookup table matrix that maps discrete tokens into continuous dense vectors. Because the embedding matrix size scales linearly with the vocabulary size, keeping a modest vocabulary (e.g., 8k–16k tokens) is vital so that this single layer does not monopolize the 16MB budget.
- **Attention (context mechanism):** The core mechanism allowing the model to weigh the relevance of past tokens when generating new ones. To respect the 16MB limit and the tight 10-minute training constraint, memory-efficient variants like **Grouped-Query Attention (GQA)** or **Multi-Query Attention (MQA)** are heavily favored over standard Multi-Head Attention to conserve parameters and speed up computation.
- **Vocabulary Size (V):** The total number of unique tokens the model can recognize. In a 16MB constrained model, this must be kept small (e.g., 8,192) to prevent the embedding matrix from eating up the entire parameter budget.
- **Hidden Dimension (D):** The size of the feature vector representing each token throughout the model layers. A larger dimension increases capacity but quadratically increases parameter count in the Attention and MLP layers.
- **Layers (L):** The number of transformer blocks (Attention + MLP) stacked sequentially. Increasing depth helps reasoning but linearly limits the possible dimension or vocabulary size within a fixed budget.
- **Attention Heads (H) & KV Heads:** The number of parallel attention mechanisms. In GQA, we use fewer Key/Value (KV) heads than Query heads to drastically reduce the size of the attention weight matrices.
- **MLP Hidden Size:** The expanded dimension inner layer of the Feed-Forward network (typically a multiple of D, like 4*D). It is where the "knowledge" of the model is stored, but it is very parameter-heavy.
- **LLM (Large Language Model):** The specific architecture of La Pulga.

- **BPE (Byte-Pair Encoding):** The tokenization algorithm that builds a shared vocabulary by iteratively merging the most frequent adjacent byte/character pairs in a corpus. A *custom-trained* BPE on the target dataset (e.g., TinyStories) is critical: using a pre-trained tokenizer (like tiktoken's cl100k_base) with a reduced vocab causes massive token corruption because any out-of-range token maps to a single fallback ID.

- **Mixed Precision Training:** The strategy of training in high-precision (FP32) for numerical stability while exporting the final weights in low-precision (FP16 or INT4) to meet an artifact size constraint. The key insight: casting to FP16 *before* training causes AdamW's second-moment accumulators to overflow (max FP16 ≈ 65,504), leading to NaN loss within ~10 batches.

- **Autoregressive Generation:** The inference loop where the model generates one token at a time, and each newly generated token is appended to the input context to predict the next token. This creates a feedback loop: each prediction depends on all previous predictions.

- **Temperature (sampling):** A scalar applied to the logits before sampling the next token. `T > 1` flattens the distribution (more random, creative output). `T < 1` sharpens it (more deterministic, repetitive output). `T = 0` becomes greedy search (always picks the top-1 token).

- **KV Cache:** An optimization for autoregressive inference that caches the Key and Value tensors computed for all previous tokens in the Attention layer. Without it, every new token requires recomputing attention over the full context from scratch, making generation O(n²) in sequence length.

- **Epoch:** One complete pass of the training loop over the entire dataset. Multiple epochs allow the model to revisit and reinforce patterns. Trade-off: too many epochs leads to memorization (overfitting) rather than generalization.

- **Validation Perplexity:** Perplexity measured on a held-out set the model has never seen during training. This is the definitive quality score reported in the Parameter Golf leaderboard format — not train loss. Lower is better.

- **Weight Tying:** The technique of sharing the same weight matrix between the input Embedding layer and the output projection (logit) layer. Since both map between the same token space and vector space, tying them cuts ~2M parameters from the budget and often improves training stability.
