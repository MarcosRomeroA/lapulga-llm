# RunPod H100 Setup Guide — lapulga-llm

Single-GPU iteration workflow on a RunPod 1x H100 PCIe instance using the official OpenAI Parameter Golf template.

---

## 1. Prerequisites: SSH Key

If you don't have an SSH key yet:

```bash
ssh-keygen -t ed25519 -C "runpod" -f ~/.ssh/id_runpod
```

Add the **public key** to RunPod:
1. Go to **RunPod Settings → SSH Public Keys**
2. Paste the contents of `~/.ssh/id_runpod.pub`

---

## 2. Deploy the Pod

Use the official Parameter Golf template:

```
https://console.runpod.io/hub/template/parameter-golf?id=y5cejece4j
```

Configuration:
- **GPU:** 1x H100 PCIe (or NVL)
- **Mode:** Spot / Interruptible *(~70% cheaper than on-demand)*
- **Container Disk:** 20 GB
- **Volume:** 50 GB *(mounts at `/workspace` — persists across restarts)*

> The template pre-installs `uv`, Python 3.12, and PyTorch 2.9.1 with CUDA 12.x. No additional setup needed.

---

## 3. Connect via VS Code Remote-SSH

### Get the SSH string

From the RunPod dashboard, open your pod and click **Connect → SSH**. You'll see something like:

```
ssh root@213.173.105.12 -p 14823 -i ~/.ssh/id_runpod
```

### Configure `~/.ssh/config` for easy access

```
Host runpod-h100
    HostName 213.173.105.12
    Port     14823
    User     root
    IdentityFile ~/.ssh/id_runpod
```

Replace the IP and port with the values from your pod dashboard each time you start a new pod.

### Attach VS Code

1. Install the **Remote - SSH** extension in VS Code
2. Press `Ctrl+Shift+P` → **Remote-SSH: Connect to Host**
3. Select `runpod-h100`
4. Open `/workspace` as your working folder

---

## 4. Project Setup on the Pod

Run these commands once after connecting. Everything goes into `/workspace` (the persistent volume).

```bash
# Clone the repo
cd /workspace
git clone https://github.com/YOUR_USERNAME/lapulga-llm.git
cd lapulga-llm

# Install dependencies (uv already available on the template)
uv sync

# Download FineWeb sp1024 dataset (10 training shards + validation)
# Stores data in /workspace/lapulga-llm/data/datasets/fineweb10B_sp1024/
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
```

> The dataset download takes ~10 minutes and stores ~20 GB in the persistent volume.
> You only need to do this once per volume — it survives pod restarts.

---

## 5. Iteration Workflow

### Fast validation (DEV_MODE — 100 steps, ~3 min on H100)

```bash
DEV_MODE=1 uv run main.py
```

Verifies:
- No OOM errors
- `torch.compile` succeeds (first step slow, rest fast)
- LR warmup visible in logs
- Artifact size printed at the end

### Full training run (~30–50 min on H100)

```bash
uv run main.py
```

### Run compliance tests

```bash
uv run pytest tests/test_spec_compliance.py -v
```

### Generate text from a saved checkpoint

```bash
make generate PROMPT="The quick brown fox"
```

---

## 6. Cost Management

> **IMPORTANT:** Always **STOP** (not terminate) the pod when you're not actively training.

- **Stop** = pod goes to sleep, volume persists, you pay only for storage (~$0.20/day for 50 GB)
- **Terminate** = everything deleted including the volume

From the RunPod dashboard: click the **■ Stop** button (not the trash icon).

Spot instances can be reclaimed by RunPod at any time. For the final full run, consider switching to **On-Demand** mode to avoid interruption.

---

## 7. Architecture Note (H100 vs RTX 3090)

The Fat ALBERT architecture (`dim=768`, `physical_layers=4`, `repeat_count=3`) was designed specifically for H100:

| GPU | L1 Shared Mem/SM | dim=768 Triton RMSNorm backward | Result |
|:---|:---|:---|:---|
| RTX 3090 | 101 KB | ~170 KB required | OOM |
| H100 PCIe | 228 KB | ~170 KB required | OK |

This is why `DEV_MODE=1 uv run main.py` locally (RTX 3090) will OOM — it's expected. The final run targets the H100 exclusively.
