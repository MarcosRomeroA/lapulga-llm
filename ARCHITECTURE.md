# lapulga-llm: Project Architecture & Technical Standards

## 🐐 The "La Pulga" Philosophy
Efficiency is king. Every line of code written in this project must prioritize:
1. **Memory Management:** Extreme diligence to strictly stay under the critical 16MB artifact boundary.
2. **Compute Speed:** Zero overhead abstraction to ensure the model can train from scratch in under the 10-minute time constraint on 8xH100s.

## 🧱 Modular Design
To prevent spaghetti code, the repository must be strictly organized into the following logical modules:
- **/src/data**: Data loading, dataset streaming, and Tokenization pipelines.
- **/src/model**: Transformer block architecture (MLX/PyTorch) and Weight initialization.
- **/src/training**: Optimization loops (AdamW), gradient updates, and Loss/Perplexity tracking.
- **/src/utils**: Weight quantization routines, file size calculators, and benchmarking scripts.

## 🏛️ Clean Architecture (Separation of Concerns)
Beyond just writing readable syntax, the project strictly enforces **Clean Architecture** patterns:
- **Domain Entities (Inner Ring):** Business rules and definition schemas (e.g., `ModelConfig`) must be pure, vanilla Python `dataclasses`. They must **NOT** import or depend on frameworks like `mlx` or `torch`.
- **Dependency Inversion:** Outer rings (Frameworks, CLIs, Tensors) must depend on the inner rings (Configs, Abstract Interfaces), never the other way around. 
- **Framework Isolation:** MLX-specific tensor operations or PyTorch wrappers must be strictly isolated to their own specific modules (e.g., `/src/model/mlx_backend`). The core training orchestrator should ideally not care if it's feeding arrays to a Mac or an H100.

## 🧹 Clean Code Standards
- **PEP8 Compliance:** All Python code must respect standard PEP8 styling.
- **Descriptive Naming:** Variables must be highly semantic and descriptive. Generic variable names like `x`, `y`, or `data` are **strictly prohibited** (unless tracking standard mathematical tensors where $x$ and $y$ are universally accepted, though `input_tensor` and `target_tokens` are always preferred).
- **Single Responsibility Principle (SRP):** Every function or class must execute exactly one logical task. 

## 🏷️ Type Hinting
- **Strict Typing:** Python type hints (e.g., `def forward(self, input_ids: mx.array) -> mx.array:`) must be enforced for all function signatures. This drastically improves debugging predictability and tensor shape tracking across M4/MLX.

## 📝 Documentation
- **Docstrings:** Every module, class, and function must include a clear docstring.
- **Explain the "What" and "Why":** The docstring must explicitly explain *What* the code does, and exactly *Why* it exists in the codebase (e.g., "Why did we choose this hyperparameter?").

---
**🤖 Agent Directive:** The AI assistant (*Antigravity*) acting on this repository is strictly commanded to **evaluate and reject** any code suggestion or draft that fails to adhere to these modular, type-hinted, and clean-code standards. All future code must be refactored to align with this file before being proposed.
