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
    print(f"[orchestrator] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Step 1: Start Pod
# ---------------------------------------------------------------------------
log(f"Resuming pod {POD_ID}...")
try:
    runpod.resume_pod(pod_id=POD_ID, gpu_count=8)
except runpod.error.QueryError as e:
    if "spot pod" in str(e).lower():
        log("Spot pod detected. Using podBidResume via GraphQL...")
        from runpod.api.graphql import run_graphql_query
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
        run_graphql_query(query)
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
# Step 3: Wait for SSH daemon to accept connections
# ---------------------------------------------------------------------------
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
    # Will hit the finally block to stop the pod
    raise RuntimeError("SSH daemon did not become ready within the timeout.")

# ---------------------------------------------------------------------------
# Steps 4–6: Training, artifact download, and guaranteed pod stop
# ---------------------------------------------------------------------------
SSH_OPTS = [
    "-i", SSH_KEY,
    "-p", str(ssh_port),
    "-o", "StrictHostKeyChecking=no",
    "-o", "BatchMode=yes",
    "-t",
]
REMOTE = f"root@{ssh_ip}"

try:
    # ---- Execute training ----
    log("Starting remote training...")
    subprocess.run(
        ["ssh"] + SSH_OPTS + [
            REMOTE,
            "if [ ! -d '/workspace/lapulga-llm' ]; then git clone https://github.com/MarcosRomeroA/lapulga-llm.git /workspace/lapulga-llm; fi && "
            "cd /workspace/lapulga-llm && git pull && "
            "RUN_ID=baseline_sp1024 "
            "DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024/ "
            "TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model "
            "VOCAB_SIZE=1024 "
            "PYTHONUNBUFFERED=1 "
            "torchrun --standalone --nproc_per_node=8 train_gpt.py",
        ],
        check=True,
    )
    log("Training completed successfully.")

    # ---- Download artifacts ----
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_dir = os.path.join("records", timestamp)
    os.makedirs(dest_dir, exist_ok=True)
    log(f"Downloading artifacts to {dest_dir}/")

    subprocess.run(
        ["scp"] + ["-i", SSH_KEY, "-P", str(ssh_port), "-o", "StrictHostKeyChecking=no"] + [
            f"{REMOTE}:/workspace/lapulga-llm/lapulga_submission_bf16.pt",
            f"{dest_dir}/lapulga_submission_bf16.pt",
        ],
        check=True,
    )
    log("Downloaded lapulga_submission_bf16.pt")

    # Training log is optional — don't fail if absent
    result = subprocess.run(
        ["scp"] + ["-i", SSH_KEY, "-P", str(ssh_port), "-o", "StrictHostKeyChecking=no"] + [
            f"{REMOTE}:/workspace/lapulga-llm/training_run_oficial.log",
            f"{dest_dir}/training_run_oficial.log",
        ],
    )
    if result.returncode == 0:
        log("Downloaded training_run_oficial.log")
    else:
        log("training_run_oficial.log not found on remote (skipping).")

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
