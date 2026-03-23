"""
train_gpt.py — La Pulga: Challenger Edition
OpenAI Parameter-Golf submission format: single standalone file.
Optimized for 8xH100 SXM, 600s wallclock, DDP, and 16MB int8+zlib limit.
"""

from __future__ import annotations

import glob
import io
import math
import os
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP


# -----------------------------
# HYPERPARAMETERS & CONFIG
# -----------------------------

class Hyperparameters:
    # Environment & Paths
    data_path = os.environ.get("DATA_PATH", "/workspace/data/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Architecture (Target ~8.1M params to fit 16MB int8+zlib limit)
    # Using the user-requested custom config: 10 layers, 8 heads, 256 dim
    dim = 256
    physical_layers = 10
    n_heads = 8
    n_kv_heads = 4
    hidden_dim = 1536  # ~6x expansion for parameters, or maybe keep 3x = 768. Let's use 768 to be safe on size
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    context_length = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    logit_softcap = 30.0

    # Training Loop
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 590.0))  # 10s margin for save
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 500))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 50))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))

    # Optimizer (AdamW)
    learning_rate = 6e-4
    matrix_lr = 0.02
    muon_momentum = 0.95
    muon_steps = 5
    beta1 = 0.9
    beta2 = 0.95
    weight_decay = 0.1
    grad_clip_norm = 1.0


# -----------------------------
# MUON OPTIMIZER
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, weight_decay=weight_decay),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            wd = group.get("weight_decay", 0.0)
            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                if wd > 0:
                    p.data.mul_(1.0 - lr * wd)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION
# -----------------------------

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
        
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    model.eval()
    
    # We do a simplified single-pass evaluation
    # Split tokens among ranks
    total_seqs = (val_tokens.numel() - 1) // args.context_length
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    
    local_batch_seqs = 32
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.context_length
            raw_end = batch_seq_end * args.context_length + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            
            x = local[:-1].reshape(-1, args.context_length)
            y = local[1:].reshape(-1, args.context_length)
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                batch_loss = model(x, y).detach()
                
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
            
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
        
    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


# -----------------------------
# DATA LOADING
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

def load_validation_tokens(args: Hyperparameters) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(args.val_files))]
    if not files:
        raise FileNotFoundError(f"No validation files found for pattern: {args.val_files}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // args.context_length) * args.context_length
    return tokens[: usable + 1]


class DistributedTokenLoader:
    """
    Loads training tokens sequentially. In DDP, each rank loads only its portion
    of the sequence to avoid redundant memory and CPU processing.
    """
    def __init__(self, pattern: str, seq_len: int, batch_tokens: int, rank: int, world_size: int):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No training files found for {pattern}")
            
        self.seq_len = seq_len
        self.micro_batch_seqs = (batch_tokens // seq_len) // world_size
        self.rank_tokens = self.micro_batch_seqs * seq_len
        
        self.rank = rank
        self.world_size = world_size
        self.file_idx = 0
        self.pos = 0
        self._load_next_shard()

    def _load_next_shard(self):
        self.tokens = load_data_shard(self.files[self.file_idx]).to(torch.int32)
        self.pos = self.rank * self.rank_tokens
        self.file_idx = (self.file_idx + 1) % len(self.files)

    def next_batch(self, device: torch.device) -> tuple[Tensor, Tensor]:
        need = self.rank_tokens + 1
        chunks = []
        
        while need > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._load_next_shard()
                continue
            take = min(need, avail)
            chunks.append(self.tokens[self.pos : self.pos + take])
            self.pos += take * self.world_size  # stride by world size
            need -= take
            
        chunk = torch.cat(chunks)
        x = chunk[:-1].reshape(-1, self.seq_len).to(device=device, dtype=torch.int64, non_blocking=True)
        y = chunk[1:].reshape(-1, self.seq_len).to(device=device, dtype=torch.int64, non_blocking=True)
        return x, y


# -----------------------------
# ARCHITECTURE (La Pulga)
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 1024, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos_table", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_table", freqs.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        return (
            self.cos_table[:, :, :seq_len, :].to(dtype=dtype),
            self.sin_table[:, :, :seq_len, :].to(dtype=dtype),
        )

def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class Attention(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        kv_dim = self.n_kv_heads * self.head_dim

        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, kv_dim, bias=False)
        self.wv = nn.Linear(args.dim, kv_dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)
        nn.init.zeros_(self.wo.weight)

        self.q_gain = nn.Parameter(torch.full((self.n_heads,), 1.0, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, max_seq_len=args.context_length)

    def forward(self, x: Tensor) -> Tensor:
        b, s, _ = x.shape

        q = self.wq(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        cos, sin = self.rotary(s, dtype=q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=(self.n_kv_heads != self.n_heads)
        )
        return self.wo(out.transpose(1, 2).contiguous().reshape(b, s, -1))


class MLP(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.fc = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.proj = nn.Linear(args.hidden_dim, args.dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(torch.relu(self.fc(x)).square())


class TransformerBlock(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = MLP(args)
        self.attention_norm = RMSNorm()
        self.ffn_norm = RMSNorm()
        self.attn_scale = nn.Parameter(torch.ones(args.dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(args.dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        attn_out = self.attention(self.attention_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.feed_forward(self.ffn_norm(x))
        return x


class LanguageModel(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.config = args
        
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=0.02)
        
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.physical_layers)])
        self.norm = RMSNorm()

    def forward(self, input_ids: Tensor, target_ids: Tensor | None = None) -> Tensor:
        x = self.tok_embeddings(input_ids)
        x = F.rms_norm(x, (x.size(-1),))

        for block in self.layers:
            x = block(x)

        x = self.norm(x)
        logits = F.linear(x, self.tok_embeddings.weight)
        logits = self.config.logit_softcap * torch.tanh(logits / self.config.logit_softcap)

        if target_ids is not None:
            return F.cross_entropy(
                logits.float().reshape(-1, self.config.vocab_size),
                target_ids.reshape(-1),
                reduction="mean",
            )
        return logits


# -----------------------------
# MAIN LOOP
# -----------------------------

def main():
    args = Hyperparameters()
    
    # DDP Setup
    distributed = "RANK" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group("nccl", device_id=device)
        
    master_process = rank == 0

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed + rank)

    if master_process:
        print(f"--- DDP Initialized: {world_size} GPUs ---")
        
    model = LanguageModel(args).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    
    if master_process:
        print(f"Model parameters: {num_params:,}")
        
    if distributed:
        model = DDP(model, device_ids=[local_rank])
    model_unwrapped = model.module if distributed else model

    # Optimizer: Muon for 2D params, AdamW for 1D params/embeddings
    # Fast iterations want FP32 1D params
    muon_params = []
    adam_params = []
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim < 2 or "tok_embeddings" in name:
                param.data = param.data.float()
                adam_params.append(param)
            else:
                muon_params.append(param)

    optimizer_muon = Muon(
        muon_params, 
        lr=args.matrix_lr, 
        momentum=args.muon_momentum, 
        backend_steps=args.muon_steps, 
        weight_decay=args.weight_decay
    )
    
    optimizer_adam = torch.optim.AdamW(
        adam_params, 
        lr=args.learning_rate, 
        betas=(args.beta1, args.beta2), 
        weight_decay=args.weight_decay,
        fused=True
    )
    
    # Load evaluation utilities
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    base_bytes, has_space, is_boundary = build_sentencepiece_luts(sp, args.vocab_size, device)
    val_tokens = load_validation_tokens(args)
    if master_process:
        print(f"Loaded {val_tokens.numel():,} validation tokens")

    # Load dataloader
    train_loader = DistributedTokenLoader(
        args.train_files, args.context_length, args.train_batch_tokens, rank, world_size
    )

    # Compile model 
    if master_process:
        print("Compiling model (mode=max-autotune)...")
    model = torch.compile(model, mode="max-autotune")
    
    # Pre-fetch the first batch
    x, y = train_loader.next_batch(device)

    # Start timer!
    t0_wallclock = time.perf_counter()
    max_wallclock = args.max_wallclock_seconds

    # Infinite training loop until wallclock triggers
    step = 0
    while True:
        # Check wallclock: Break at 590s
        elapsed = time.perf_counter() - t0_wallclock
        if elapsed >= max_wallclock:
            if master_process:
                print(f"\n--- Wallclock limit reached ({elapsed:.1f}s >= {max_wallclock}s). Halting. ---")
            break
            
        t0_step = time.perf_counter()
        
        # Forward & Backward
        x, y = train_loader.next_batch(device)
        optimizer_muon.zero_grad(set_to_none=True)
        optimizer_adam.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
            
        loss.backward()
        
        # Grad clipping
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(adam_params, args.grad_clip_norm)
            
        optimizer_muon.step()
        optimizer_adam.step()
        
        t1_step = time.perf_counter()
        
        # Logging
        if step % args.train_log_every == 0 and master_process:
            step_time = (t1_step - t0_step) * 1000
            print(f"Step {step:05d} | Time {elapsed:5.1f}s | Loss {loss.item():.4f} | dt {step_time:.1f}ms")
            
        # Evaluation
        if step > 0 and step % args.val_loss_every == 0:
            val_loss, val_bpb = eval_val(
                args, model, rank, world_size, device, val_tokens, base_bytes, has_space, is_boundary
            )
            if master_process:
                print(f"--- Eval @ Step {step} | Val Loss: {val_loss:.4f} | Val BPB: {val_bpb:.4f} ---")
                
        step += 1

        # Pre-fetch next batch before measuring time of the NEXT step
        x, y = train_loader.next_batch(device)

    # Final Eval & Save (Master only)
    if master_process:
        print("\n--- Final Evaluation ---")
        val_loss, val_bpb = eval_val(
            args, model, 0, 1, device, val_tokens, base_bytes, has_space, is_boundary
        )
        print(f"Final Val Loss: {val_loss:.4f} | Final Val BPB: {val_bpb:.4f}")
        
        print("\n--- Saving Artifact (bfloat16 direct) ---")
        state_dict = getattr(model_unwrapped, "_orig_mod", model_unwrapped).state_dict()
        
        # Convert all weights to bfloat16 to fit in 16MB limit
        bf16_state_dict = {k: v.cpu().bfloat16() for k, v in state_dict.items()}
        
        save_path = "lapulga_submission_bf16.pt"
        torch.save(bf16_state_dict, save_path)
        
        model_bytes = os.path.getsize(save_path)
        print(f"Saved {save_path}. Size: {model_bytes:,} bytes ({model_bytes / 1e6:.2f} MB)")
        
        code_bytes = len(Path(__file__).read_bytes())
        total = code_bytes + model_bytes
        print(f"Total submission size: {total:,} bytes / 16,000,000 limit")
        if total <= 16_000_000:
            print("PASS! Under 16MB limit.")
        else:
            print("FAIL! Over 16MB limit.")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
