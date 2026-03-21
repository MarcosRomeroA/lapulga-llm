# AGENTS.md - lapulga-llm Project Context & Rules

## 🎯 Primary Goal
Develop and train **lapulga-llm** (La Pulga, named in honor of Lionel Messi for being exceptionally small but mighty), a high-performance Large Language Model (LLM) that fits within a **16MB** artifact limit, trained in under **10 minutes** using **8× NVIDIA H100 GPUs**.

## 🛠️ Tech Stack & Environment
- **Local Development:** MacBook Air M4. Use **MLX** framework for hardware-accelerated prototyping on Apple Silicon.
- **Cloud Infrastructure:** Runpod (8×H100). Use **PyTorch** with **FlashAttention-2** and **Unsloth** for maximum training throughput.
- **Precision target:** Aim for **4-bit quantization** to maximize the parameter count within the 16MB budget.

## 📏 Hard Constraints
- **Artifact Size:** Maximum $16,777,216$ bytes ($16$ MiB).
- **Training Time:** Absolute limit of **10 minutes**.
- **Architecture:** Decoder-only Transformer (Llama/Gemma-style).

## 🧮 Parameter Budget Estimates
Total capacity based on weight precision:
- **16-bit (bf16/fp16):** ~8.3M parameters.
- **8-bit (int8):** ~16.7M parameters.
- **4-bit (int4):** ~33.5M parameters (**Priority strategy**).

## 📋 Architectural Guidelines for AI Agents
🚨 **CRITICAL DIRECTIVE:** The AI Agent **MUST** read and strictly adhere to the technical and coding standards defined in **`ARCHITECTURE.md`**. This establishes the absolute laws for modular structure, enforcing clean code (SRP, PEP8), mandatory Type Hinting, and the "La Pulga" philosophy. 

*Baseline Parameters:*
1. **Vocabulary Size:** Keep it small (e.g., 8k to 16k tokens) to prevent the embedding matrix from consuming too much of the 16MB budget.
2. **Attention:** Use **Grouped-Query Attention (GQA)** or **Multi-Query Attention (MQA)** to reduce memory footprint and increase inference/training speed.
3. **Layer Depth:** Experiment with 6-12 layers. Prioritize "width" (embedding dimension) over "depth" if perplexity plateaus.
4. **Data:** Focus on high-quality synthetic datasets like *TinyStories* or *Cosmopedia*. Data cleaning and filtering are more important than volume given the 10-minute window.

## 📚 Knowledge Management & Documentation

The project must follow a structured documentation approach in a `/docs` folder to track progress and technical findings.

### Folder Structure: Define a `/docs` directory.

### File Definitions:
- **`journal.md`**: Daily log of decisions, experiments, and progress.
- **`til.md` (Today I Learned)**: A chronological log of technical discoveries. **Crucial:** Treat this as a staging area. Once a finding is validated as a "project law," move it to `AGENTS.md` and archive the note.
- **`benchmarks.md`**: Quantitative tracking of model configurations (layers, dims, heads) vs. final size in bytes and perplexity scores.
- **`resources.md`**: Links to research papers, repositories (MLX, nanoGPT, Unsloth), and competition updates.
- **`glossary.md`**: Definitions of key technical terms (Quantization, GQA, MQA, Perplexity, etc.).

## 🤖 Instructions for the Agent
- **Execution Policy:** DO NOT run terminal commands directly unless explicitly requested by the user for debugging purposes. Treat the user as the principal operator and provide them the clean command instructions to run manually.
- **Architecture Strictness:** You must ALWAYS follow the standards defined in `ARCHITECTURE.md`. You are instructed to actively reject any monolithic, non-typed, or poorly named code suggestions.
- **Priority:** Learning & Transparency.
- **Code Suggestions:** Every code suggestion must include a "Why it works" section and a weight calculation.
- Whenever proposing architectural changes, **always calculate the estimated file size** in bytes.
- Prioritize code that is compatible with **MLX** for local testing.
- Be aggressive with optimization. We are 3 days into the challenge; focus on iterative, fast-to-train baselines.
- When you are asked for a technical explanation or you solve a bug, remind the user to document it in the corresponding `docs/` file.
- Ensure all documentation remains in English for international consistency.
- Maintain the focus on the 16MB limit and M4/MLX optimization in every documented finding.