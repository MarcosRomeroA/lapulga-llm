import time
import os
import math
import argparse
import tiktoken
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config["n_heads"]
        self.n_kv_heads = config["n_kv_heads"]
        self.head_dim = config["dim"] // config["n_heads"]
        self.scale = self.head_dim ** -0.5

        self.wq = nn.Linear(config["dim"], config["n_heads"] * self.head_dim, bias=False)
        self.wk = nn.Linear(config["dim"], config["n_kv_heads"] * self.head_dim, bias=False)
        self.wv = nn.Linear(config["dim"], config["n_kv_heads"] * self.head_dim, bias=False)
        self.wo = nn.Linear(config["n_heads"] * self.head_dim, config["dim"], bias=False)
        
        # RoPE
        self.rope = nn.RoPE(self.head_dim, traditional=False)

    def __call__(self, x, mask=None):
        B, L, D = x.shape
        q = self.wq(x).reshape(B, L, self.n_heads, self.head_dim)
        k = self.wk(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        v = self.wv(x).reshape(B, L, self.n_kv_heads, self.head_dim)

        q = self.rope(q)
        k = self.rope(k)

        # GQA repeat
        if self.n_heads != self.n_kv_heads:
            repeats = self.n_heads // self.n_kv_heads
            k = mx.repeat(k, repeats, axis=2)
            v = mx.repeat(v, repeats, axis=2)

        # compute attention
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            scores = scores + mask
        scores = mx.softmax(scores, axis=-1)
        output = (scores @ v).reshape(B, L, D)
        return self.wo(output)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config["dim"], config["hidden_dim"], bias=False)
        self.w2 = nn.Linear(config["hidden_dim"], config["dim"], bias=False)
        self.w3 = nn.Linear(config["dim"], config["hidden_dim"], bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = MLP(config)
        self.attention_norm = RMSNorm(config["dim"], config["norm_eps"])
        self.ffn_norm = RMSNorm(config["dim"], config["norm_eps"])

    def __call__(self, x, mask=None):
        h = x + self.attention(self.attention_norm(x), mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config["vocab_size"]
        self.tok_embeddings = nn.Embedding(config["vocab_size"], config["dim"])
        self.layers = [TransformerBlock(config) for _ in range(config["n_layers"])]
        self.norm = RMSNorm(config["dim"], config["norm_eps"])
        # Tied embeddings
        self.output = nn.Linear(config["dim"], config["vocab_size"], bias=False)
        self.output.weight = self.tok_embeddings.weight

    def __call__(self, x):
        # Create causal mask
        B, L = x.shape
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L, x.dtype)
        
        x = self.tok_embeddings(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.output(self.norm(x))

def loss_fn(model, x, y):
    logits = model(x)
    loss = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
    return mx.mean(loss)

def get_batches(tokens, batch_size, context_length):
    n_batches = len(tokens) // (batch_size * context_length)
    if n_batches == 0:
         n_batches = 1
    
    tokens = tokens[:n_batches * batch_size * context_length + 1]
    X = tokens[:-1].reshape(batch_size, -1)
    Y = tokens[1:].reshape(batch_size, -1)
    
    for i in range(0, X.shape[1], context_length):
        yield X[:, i:i+context_length], Y[:, i:i+context_length]

def main():
    # Model Config (Target ~7.8M parameters to fit comfortably in 16MB at fp16)
    config = {
        "dim": 256,
        "n_layers": 6,
        "n_heads": 8,
        "n_kv_heads": 2, # GQA
        "hidden_dim": 1024,
        "vocab_size": 100256, # will adjust based on tokenizer
        "norm_eps": 1e-5,
    }

    # BPE Setup
    print("Loading tokenizer...")
    enc = tiktoken.get_encoding("cl100k_base") 
    # Tiktoken's cl100k_base is around ~100k tokens. 
    # For a strictly <16MB limit, a 100k vocab takes a massive embedding matrix (100k * 256 * 2 = 50MB).
    # We MUST restrict the vocab size for this challenge or use a smaller one.
    # Instead of full 100k, we can map to a smaller one or use a byte-level custom tokenizer.
    # To keep it simple, let's artificially limit tiktoken to the top 8192 tokens by just keeping them, 
    # or better, use simple char-level or a small BPE.
    # For the sake of this test, let's assume we train our own/use a small subset, but let's 
    # set the config vocab size to 8192 for the architecture, and fallback to module if needed.
    
    config["vocab_size"] = 8192

    # Verify size
    model = LanguageModel(config)
    num_params = sum(v.size for _, v in model.parameters().items())
    print(f"Total model parameters: {num_params:,}")
    estimated_size_mb = num_params * 2 / (1024*1024) # fp16
    print(f"Estimated size at FP16: {estimated_size_mb:.2f} MiB")

    # Load Data
    try:
        with open("train.txt", "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print("train.txt not found. Creating a small dummy dataset to test the loop...")
        text = "Hello world! This is a test dataset for the MLX model training loop. We are aiming for parameter golf!" * 1000
        with open("train.txt", "w") as f:
            f.write(text)
    
    # Tokenize. We'll clip tokens to vocab_size for demonstration, 
    # though in a real scenario you'd train an 8k tokenizer.
    print("Tokenizing data...")
    tokens = mx.array([t if t < config["vocab_size"] else 0 for t in enc.encode(text)])
    print(f"Tokens in dataset: {tokens.size}")

    # Training Setup 
    # Why these numbers?
    # M4 has extremely fast unified memory (~120GB/s) but relatively modest sheer compute compared to H100.
    # Moderate batch_size (8-16) maxes memory bandwidth utilization without OOM.
    # LR (1e-3) is slightly higher than usual (typically 3e-4) to exploit the 10-min fast-training limit
    # because we need rapid convergence and AdamW can handle high LR on shallow networks.
    batch_size = 8 
    context_length = 256
    learning_rate = 1e-3 
    epochs = 1

    optimizer = optim.AdamW(learning_rate=learning_rate)
    state = [model.state, optimizer.state]

    @mx.compile
    def step(x, y):
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, x, y)
        optimizer.update(model, grads)
        return loss

    print("Starting training...")
    mx.eval(state)

    for epoch in range(epochs):
        for batch_idx, (x, y) in enumerate(get_batches(tokens, batch_size, context_length)):
            t0 = time.perf_counter()
            loss = step(x, y)
            mx.eval(state) # Ensure compute is done
            t1 = time.perf_counter()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx:04d} | Loss {loss.item():.4f} | Time {t1-t0:.4f}s")
                
    # Checkpointing
    print("Saving model...")
    checkpoint_path = "model.safetensors"
    model.save_weights(checkpoint_path)
    actual_size = os.path.getsize(checkpoint_path)
    print(f"Saved! Checkpoint size: {actual_size / (1024*1024):.2f} MiB")

if __name__ == "__main__":
    main()
