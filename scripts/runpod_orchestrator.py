#!/usr/bin/env python3
"""
Zero-Click RunPod MLOps Orchestrator
-------------------------------------
Starts a stopped RunPod Pod, waits for SSH, runs training,
downloads artifacts, and guarantees the Pod is stopped afterward.

Required environment variables:
  RUNPOD_API_KEY       - RunPod API key
  RUNPOD_POD_ID        - ID of the existing stopped Pod

Optional environment variables:
  RUNPOD_SSH_KEY_PATH  - Path to SSH private key (default: ~/.ssh/id_rsa)
"""
import datetime
import os
import socket
import subprocess
import sys
import time

import requests
import runpod

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("RUNPOD_API_KEY")
POD_ID = os.environ.get("RUNPOD_POD_ID")
SSH_KEY = os.environ.get("RUNPOD_SSH_KEY_PATH", os.path.expanduser("~/.ssh/id_rsa"))

if not API_KEY:
    sys.exit("ERROR: RUNPOD_API_KEY environment variable is not set.")
if not POD_ID:
    sys.exit("ERROR: RUNPOD_POD_ID environment variable is not set.")

runpod.api_key = API_KEY


def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}][orchestrator] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Step 1: Start Pod
# ---------------------------------------------------------------------------
log(f"Resuming pod {POD_ID}...")
try:
    runpod.resume_pod(pod_id=POD_ID, gpu_count=8)
except Exception as e:
    if "spot" in str(e).lower() or "bid" in str(e).lower():
        log("Spot pod detected. Using podBidResume via GraphQL...")
        pod_info = runpod.get_pod(POD_ID)
        gpu_count = pod_info.get("gpuCount", 8)
        bid = float(os.environ.get("RUNPOD_SPOT_BID", pod_info.get("costPerHr") or 0.2))
        query = f"""
        mutation {{
            podBidResume(input: {{ podId: "{POD_ID}", gpuCount: {gpu_count}, bidPerGpu: {bid} }}) {{
                id
                desiredStatus
            }}
        }}
        """
        resp = requests.post(
            "https://api.runpod.io/graphql",
            json={"query": query},
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=30,
        )
        resp.raise_for_status()
    else:
        raise
log("Resume request sent.")

# ---------------------------------------------------------------------------
# Step 2: Poll until RUNNING and extract SSH info
# ---------------------------------------------------------------------------
SSH_POLL_INTERVAL = 10
SSH_POLL_MAX = 60  # 10 minutes max

ssh_ip = None
ssh_port = None

for attempt in range(SSH_POLL_MAX):
    time.sleep(SSH_POLL_INTERVAL)
    pod = runpod.get_pod(POD_ID)
    status = pod.get("desiredStatus") or pod.get("status", "")
    runtime = pod.get("runtime")
    log(f"Poll {attempt + 1}/{SSH_POLL_MAX}: status={status}, runtime={'yes' if runtime else 'no'}")

    if status == "RUNNING" and runtime:
        ports = runtime.get("ports") or []
        for p in ports:
            if p.get("privatePort") == 22:
                ssh_ip = p.get("ip")
                ssh_port = p.get("publicPort")
                break
        if ssh_ip and ssh_port:
            log(f"Pod is RUNNING. SSH at {ssh_ip}:{ssh_port}")
            break
else:
    sys.exit("ERROR: Pod did not reach RUNNING state within the timeout.")

# ---------------------------------------------------------------------------
# Steps 3–6: SSH ready check, training, artifact download + guaranteed stop
# ---------------------------------------------------------------------------
SSH_OPTS = [
    "-i", SSH_KEY,
    "-p", str(ssh_port),
    "-o", "StrictHostKeyChecking=no",
    "-o", "BatchMode=yes",
]
REMOTE = f"root@{ssh_ip}"

try:
    # ---- Wait for SSH daemon to accept connections ----
    SSH_READY_RETRIES = 60
    SSH_READY_INTERVAL = 5

    log("Waiting for SSH daemon to become ready...")
    for attempt in range(SSH_READY_RETRIES):
        try:
            conn = socket.create_connection((ssh_ip, ssh_port), timeout=3)
            conn.close()
            log("SSH daemon is accepting connections.")
            break
        except (socket.timeout, ConnectionRefusedError, OSError):
            log(f"SSH not ready yet (attempt {attempt + 1}/{SSH_READY_RETRIES}), retrying in {SSH_READY_INTERVAL}s...")
            time.sleep(SSH_READY_INTERVAL)
    else:
        raise RuntimeError("SSH daemon did not become ready within the timeout.")

    # ---- Upload training script ----
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(project_root, "records", "lapulga-llm", "train_gpt.py")
    if not os.path.exists(script_path):
        subprocess.run(["python", os.path.join(project_root, "scripts", "build_submission.py")], check=True)
    log(f"Uploading local script to remote: {script_path}")
    subprocess.run(
        ["scp"] + ["-i", SSH_KEY, "-P", str(ssh_port), "-o", "StrictHostKeyChecking=no"] + [
            script_path,
            f"{REMOTE}:/workspace/train_gpt.py",
        ],
        check=True,
    )

    # ---- Prepare local run directory (used for log + artifacts) ----
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_dir = os.path.join("records", timestamp)
    os.makedirs(dest_dir, exist_ok=True)
    local_log_path = os.path.join(dest_dir, "orchestrator.log")

    # ---- Execute training — stream output line-by-line to terminal + local log ----
    TRAINING_TIMEOUT_SECS = 11 * 60  # hard cut at 11:00 min (competition limit is 10:00)
    log(f"Starting remote training. Live output → {local_log_path}")
    print("=" * 72, flush=True)

    training_returncode = None
    proc = subprocess.Popen(
        ["ssh"] + SSH_OPTS + [
            REMOTE,
            "echo '[step 1/4] checking parameter-golf repo...' && "
            "if [ ! -d '/workspace/parameter-golf/.git' ]; then "
            "  echo '[step 1/4] cloning parameter-golf...' && "
            "  rm -rf /workspace/parameter-golf && "
            "  git clone https://github.com/openai/parameter-golf.git /workspace/parameter-golf; "
            "else echo '[step 1/4] repo already present, skipping clone.'; fi && "
            "echo '[step 2/4] checking FineWeb dataset...' && "
            "if [ ! -d '/workspace/parameter-golf/data/datasets/fineweb10B_sp1024' ]; then "
            "  echo '[step 2/4] dataset not found — downloading (this may take several minutes)...' && "
            "  cd /workspace/parameter-golf && python3 data/cached_challenge_fineweb.py --variant sp1024; "
            "else echo '[step 2/4] dataset already present, skipping download.'; fi && "
            "echo '[step 3/4] installing dependencies...' && "
            "pip install --quiet safetensors sentencepiece && "
            "echo '[step 3/4] dependencies installed.' && "
            "mv /workspace/train_gpt.py /workspace/parameter-golf/train_gpt.py && "
            "echo '[step 4/4] launching torchrun (8 GPUs)...' && "
            "cd /workspace/parameter-golf && "
            "RUN_ID=baseline_sp1024 "
            "VOCAB_SIZE=1024 "
            "PYTHONUNBUFFERED=1 "
            "torchrun --standalone --nproc_per_node=8 train_gpt.py",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    with open(local_log_path, "w") as log_file:
        try:
            for line in proc.stdout:
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                out = f"[{ts}][train] {line.rstrip()}"
                print(out, flush=True)
                log_file.write(out + "\n")
                log_file.flush()
            proc.wait(timeout=TRAINING_TIMEOUT_SECS)
            training_returncode = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            training_returncode = -1
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            msg = f"[{ts}][orchestrator] TIMEOUT: training cut at {TRAINING_TIMEOUT_SECS // 60} min."
            print(msg, flush=True)
            log_file.write(msg + "\n")

    print("=" * 72, flush=True)
    if training_returncode == 0:
        log("Training completed successfully.")
    elif training_returncode == -1:
        log("Training timed out — downloading whatever artifacts exist.")
    else:
        log(f"Training exited with code {training_returncode} — attempting artifact download anyway.")

    # ---- Download artifacts (best-effort: always runs, even after partial training) ----
    log(f"Downloading artifacts to {dest_dir}/")

    SCP_BASE = ["scp"] + ["-i", SSH_KEY, "-P", str(ssh_port), "-o", "StrictHostKeyChecking=no"]

    for remote_path, local_name, required in [
        ("/workspace/parameter-golf/lapulga_submission_bf16.pt", "lapulga_submission_bf16.pt", True),
        ("/workspace/parameter-golf/training_run_oficial.log", "training_run_oficial.log", False),
    ]:
        result = subprocess.run(SCP_BASE + [f"{REMOTE}:{remote_path}", f"{dest_dir}/{local_name}"])
        if result.returncode == 0:
            log(f"Downloaded {local_name}")
        elif required:
            log(f"WARNING: {local_name} not found on remote (training may have been cut before first checkpoint).")
        else:
            log(f"{local_name} not found on remote (skipping).")

    log(f"All artifacts saved to {dest_dir}/")

finally:
    # ---- Guaranteed kill-switch ----
    log("Stopping pod (kill-switch)...")
    try:
        runpod.stop_pod(pod_id=POD_ID)
        log("Pod stopped successfully.")
    except Exception as stop_err:
        log(f"WARNING: Failed to stop pod automatically: {stop_err}")
        log(f"PLEASE MANUALLY STOP POD {POD_ID} IN THE RUNPOD CONSOLE TO AVOID CHARGES.")
