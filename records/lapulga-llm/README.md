## lapulga-llm submission

### Summary
- Track: `track_10min_16mb`
- Training target: <= 600s on 8xH100 SXM
- Artifact limit: < 16,000,000 bytes (code + compressed model)
- Current validated artifact: ~15.21MB (`int8+zlib`)

### Model
- Decoder-only Transformer with ALBERT-style sharing (4 physical layers, 12 effective passes)
- `dim=768`, `heads=12`, `kv_heads=4`, `hidden_dim=1280`, `vocab=1024`
- Weight tying + GQA + U-Net skips + logit softcap

### Run
```bash
python train_gpt.py
```

### Notes
- This folder is prepared in the same submission style as `parameter-golf/records/*`: includes `train_gpt.py`, `submission.json`, and `requirements.txt`.
