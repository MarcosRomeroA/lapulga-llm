
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from src.domain.config import ModelConfig

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no learnable weight for compile compat)."""
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class Rotary(nn.Module):
    """
    Rotary Position Embeddings with pre-computed cos/sin tables.
    """
    def __init__(self, dim: int, max_seq_len: int = 1024, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        cos = self.cos[:seq_len].unsqueeze(0).unsqueeze(2).to(x.dtype) # [1, T, 1, head_dim]
        sin = self.sin[:seq_len].unsqueeze(0).unsqueeze(2).to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class LanguageModel(nn.Module):
    """
    Banked Transformer Architecture.
    Instead of separate nn.Linear layers, weights are stacked in 3D tensors: [n_layers, out_features, in_features].
    This allows parallel orthogonalization (Parallel Muon) and minimizes Kernel Launch Overhead.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.physical_layers
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.hidden_dim = config.hidden_dim
        self.logit_softcap = config.logit_softcap
        
        # Embeddings
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        
        # Positional Encoding
        self.rotary = Rotary(self.head_dim)
        
        # 3D Parameter Banks (Parameter Banking)
        # QKV Bank: [n_layers, out_dim, in_dim]
        # out_dim = (n_heads + 2*n_kv_heads) * head_dim
        qkv_dim = (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
        self.bank_qkv = nn.Parameter(torch.empty(self.n_layers, qkv_dim, self.dim))
        
        # O Bank: [n_layers, dim, dim]
        self.bank_o = nn.Parameter(torch.empty(self.n_layers, self.dim, self.dim))
        
        # MLP Banks
        self.bank_fc = nn.Parameter(torch.empty(self.n_layers, self.hidden_dim, self.dim))
        self.bank_proj = nn.Parameter(torch.empty(self.n_layers, self.dim, self.hidden_dim))
        
        # Q-Gain (per layer)
        self.q_gain = nn.Parameter(torch.empty(self.n_layers, self.head_dim))
        
        # Residual Mix Scales
        self.attn_scale = nn.Parameter(torch.ones(self.n_layers))
        self.mlp_scale = nn.Parameter(torch.ones(self.n_layers))
        self.resid_mix = nn.Parameter(torch.ones(self.n_layers))
        
        # U-Net Skip Weights
        num_encoder = self.n_layers // 2
        num_decoder = self.n_layers - num_encoder
        self.num_skip_weights = min(num_encoder, num_decoder)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, config.dim, dtype=torch.float32))
        
        self.norm = RMSNorm()
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.bank_qkv, mean=0.0, std=0.02)
        nn.init.normal_(self.bank_o, mean=0.0, std=0.02)
        nn.init.normal_(self.bank_fc, mean=0.0, std=0.02)
        nn.init.zeros_(self.bank_proj) # zero init for projection
        nn.init.ones_(self.q_gain)

    def forward(self, input_ids: Tensor, target_ids: Optional[Tensor] = None) -> Tensor:
        B, T = input_ids.size()
        x = self.tok_embeddings(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []
        
        num_encoder = self.n_layers // 2
        decoder_step = 0
        
        for i in range(self.n_layers):
            # U-Net Skip connection
            if i >= num_encoder and skips:
                sw = self.skip_weights[decoder_step].to(dtype=x.dtype)[None, None, :]
                x = x + sw * skips.pop()
                decoder_step += 1
            elif i < num_encoder:
                skips.append(x)
                
            x_prev = x
            
            # --- Attention ---
            x_norm = F.rms_norm(x, (x.size(-1),))
            # Linear via bmm or matmul: x_norm [B, T, dim] @ bank_qkv[i].T [dim, qkv_dim]
            w_qkv = self.bank_qkv[i].to(x.dtype)
            qkv = F.linear(x_norm, w_qkv)
            
            # Split QKV
            q_dim = self.n_heads * self.head_dim
            kv_dim = self.n_kv_heads * self.head_dim
            q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
            
            q = q.view(B, T, self.n_heads, self.head_dim)
            k = k.view(B, T, self.n_kv_heads, self.head_dim)
            v = v.view(B, T, self.n_kv_heads, self.head_dim)
            
            q = self.rotary(q)
            k = self.rotary(k)
            
            # Apply q_gain
            q = q * self.q_gain[i].to(x.dtype)
            
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # XSA in last 2 layers
            is_xsa = (i >= self.n_layers - 2)
            if is_xsa:
                v_norm = F.normalize(v, dim=-1)
                try:
                    y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(self.n_kv_heads != self.n_heads))
                    y_grouped = y.view(B, self.n_kv_heads, self.n_heads // self.n_kv_heads, T, self.head_dim)
                    v_norm_expanded = v_norm.unsqueeze(2)
                    proj = (y_grouped * v_norm_expanded).sum(dim=-1, keepdim=True) * v_norm_expanded
                    y = (y_grouped - proj).view(B, self.n_heads, T, self.head_dim)
                except TypeError:
                    if self.n_kv_heads != self.n_heads:
                        rep = self.n_heads // self.n_kv_heads
                        k = k.repeat_interleave(rep, dim=1)
                        v = v.repeat_interleave(rep, dim=1)
                        v_norm = F.normalize(v, dim=-1)
                    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                    proj = (y * v_norm).sum(dim=-1, keepdim=True) * v_norm
                    y = y - proj
            else:
                try:
                    y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(self.n_kv_heads != self.n_heads))
                except TypeError:
                    if self.n_kv_heads != self.n_heads:
                        rep = self.n_heads // self.n_kv_heads
                        k = k.repeat_interleave(rep, dim=1)
                        v = v.repeat_interleave(rep, dim=1)
                    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                    
            y = y.transpose(1, 2).contiguous().view(B, T, self.dim)
            
            # Output projection
            w_o = self.bank_o[i].to(x.dtype)
            attn_out = F.linear(y, w_o)
            
            # Residual mix (Attention)
            scale_a = self.attn_scale[i].to(x.dtype)
            mix_a = self.resid_mix[i].to(x.dtype)
            x = torch.lerp(x_prev, x, mix_a) + attn_out * scale_a
            
            x_prev_mlp = x
            
            # --- MLP ---
            x_norm = F.rms_norm(x, (x.size(-1),))
            w_fc = self.bank_fc[i].to(x.dtype)
            w_proj = self.bank_proj[i].to(x.dtype)
            
            h = F.linear(x_norm, w_fc)
            h = F.leaky_relu(h, negative_slope=0.5).square()
            mlp_out = F.linear(h, w_proj)
            
            # Residual mix (MLP)
            scale_m = self.mlp_scale[i].to(x.dtype)
            x = x_prev_mlp + mlp_out * scale_m
            
            # Cross-layer scaling
            x = x * 0.85
            
        x = self.norm(x)
        logits = F.linear(x, self.tok_embeddings.weight.to(x.dtype))
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        
        if target_ids is not None:
            return F.cross_entropy(logits.float().reshape(-1, self.vocab_size), target_ids.reshape(-1), reduction="mean")
        return logits

    @property
    def device(self) -> torch.device:
        return self.tok_embeddings.weight.device

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
