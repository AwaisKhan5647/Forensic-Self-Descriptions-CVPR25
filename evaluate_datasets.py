"""
Comprehensive FSD evaluation script for all datasets.
Multi-GPU inference via process-per-GPU, progress tracking, resume capability.
"""

import os
import sys
import csv
import time
import json
import logging
import traceback
import subprocess
import multiprocessing as mp
from pathlib import Path
from datetime import datetime, timedelta
from queue import Empty

# ─── Configuration ───────────────────────────────────────────────────────────

DATASETS = {
    "AIGI_TEST": {
        "path": "/mnt/h200_dataset/kutub/datasets/AIGI/TEST",
        "label_mode": "subdir_01",
    },
    "image_eval24": {
        "path": "/mnt/h200_dataset/saad/datasets/images/image_eval24",
        "label_mode": "subdir_01",
    },
    "ReWIND": {
        "path": "/mnt/h200_dataset/saad/datasets/images/ancestree/ReWIND",
        "label_mode": "subdir_01",
    },
}

RESULTS_BASE = "/data/awais/projects/FSD/Results"
LOGS_DIR = "/data/awais/projects/FSD/logs"
GPU_IDS = [5, 6, 7]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".heif", ".heic"}

# ─── Logging setup ───────────────────────────────────────────────────────────

os.makedirs(LOGS_DIR, exist_ok=True)
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(LOGS_DIR, f"evaluation_{_ts}.log")
error_log_file = os.path.join(LOGS_DIR, f"errors_{_ts}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ]
)
error_handler = logging.FileHandler(error_log_file)
error_handler.setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
logger.addHandler(error_handler)


# ─── Dataset discovery ───────────────────────────────────────────────────────

def find_images_with_labels(dataset_path: str) -> list:
    """Recursively find all images under 0_real and 1_fake directories."""
    root = Path(dataset_path)
    samples = []

    for label_dir in root.rglob("0_real"):
        if label_dir.is_dir():
            for f in label_dir.rglob("*"):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append((str(f), 0))

    for label_dir in root.rglob("1_fake"):
        if label_dir.is_dir():
            for f in label_dir.rglob("*"):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append((str(f), 1))

    logger.info(f"Found {len(samples)} images in {dataset_path}")
    return samples


# ─── Progress tracker ────────────────────────────────────────────────────────

class ProgressTracker:
    def __init__(self, progress_file: str, dataset_name: str, total: int):
        self.file = progress_file
        self.dataset_name = dataset_name
        self.total = total
        self.processed = 0
        self.start_time = time.time()
        self.errors = []
        self.stage = "initializing"
        self._last_gpu = 0
        self._gpu_cache = ""

    def update(self, processed: int, stage: str = None):
        self.processed = processed
        if stage:
            self.stage = stage
        self._write()

    def add_error(self, msg: str):
        self.errors.append(msg)

    def _gpu_usage(self):
        if time.time() - self._last_gpu < 30:
            return self._gpu_cache
        try:
            ids = ",".join(str(g) for g in GPU_IDS)
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader", f"--id={ids}"],
                capture_output=True, text=True, timeout=5
            )
            self._gpu_cache = r.stdout.strip()
        except Exception:
            self._gpu_cache = "unavailable"
        self._last_gpu = time.time()
        return self._gpu_cache

    def _write(self):
        elapsed = time.time() - self.start_time
        rate = self.processed / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.processed) / rate if rate > 0 else 0
        eta = datetime.now() + timedelta(seconds=remaining)
        pct = 100 * self.processed / self.total if self.total > 0 else 0

        try:
            with open(self.file, "w") as f:
                f.write(f"Dataset: {self.dataset_name}\n")
                f.write(f"Stage: {self.stage}\n")
                f.write(f"Progress: {self.processed}/{self.total} ({pct:.1f}%)\n")
                f.write(f"Elapsed: {timedelta(seconds=int(elapsed))}\n")
                f.write(f"Rate: {rate:.2f} images/sec\n")
                f.write(f"ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')} ({timedelta(seconds=int(remaining))} remaining)\n")
                f.write(f"GPUs in use: {GPU_IDS}\n")
                f.write(f"GPU Usage:\n{self._gpu_usage()}\n")
                if self.errors:
                    f.write(f"\nErrors ({len(self.errors)}):\n")
                    for e in self.errors[-10:]:
                        f.write(f"  {e}\n")
        except Exception:
            pass


# ─── Resume support ──────────────────────────────────────────────────────────

def load_completed(csv_path: str) -> set:
    completed = set()
    if not os.path.exists(csv_path):
        return completed
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fp = row.get("file_path", "")
                if fp:
                    completed.add(fp)
        logger.info(f"Resuming: {len(completed)} files already done")
    except Exception as e:
        logger.warning(f"Could not load existing CSV: {e}")
    return completed


# ─── Worker process ──────────────────────────────────────────────────────────

def worker_process(gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue):
    """Worker that runs inference on a single GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Set cuDNN library path to fix CUDNN_STATUS_NOT_INITIALIZED with torch 2.6.0+cu124
    cudnn_lib = "/data/awais/anaconda/envs/fsd/lib/python3.12/site-packages/nvidia/cudnn/lib"
    existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{cudnn_lib}:{existing_ld}" if existing_ld else cudnn_lib

    try:
        import torch
        from fsd import FSDDetector
        detector = FSDDetector.load(device="cuda")
        result_queue.put(("ready", gpu_id, None))
    except Exception as e:
        result_queue.put(("error", gpu_id, f"init failed: {e}"))
        return

    while True:
        try:
            item = task_queue.get(timeout=60)
        except Empty:
            break

        if item is None:  # poison pill
            break

        file_path, gt = item
        try:
            result = detector.score(file_path)
            z = result.z_score
            prob = 1.0 / (1.0 + (2.718281828 ** (z + 2.0)))
            pred = 1 if result.is_fake else 0
            result_queue.put(("result", file_path, gt, prob, pred, None))
        except Exception as e:
            result_queue.put(("result", file_path, gt, None, None, str(e)))


# ─── Multi-GPU inference ─────────────────────────────────────────────────────

def run_inference(samples: list, output_dir: str, dataset_name: str) -> tuple:
    """Run inference using multiple GPUs with multiprocessing."""
    csv_path = os.path.join(output_dir, "predictions.csv")
    progress_file = os.path.join(output_dir, "progress.txt")

    completed = load_completed(csv_path)
    pending = [(p, l) for p, l in samples if p not in completed]
    total = len(samples)

    tracker = ProgressTracker(progress_file, dataset_name, total)
    tracker.processed = len(completed)
    tracker.update(len(completed), "starting")

    logger.info(f"Total: {total} | Done: {len(completed)} | Pending: {len(pending)}")

    if not pending:
        logger.info("All files already processed. Skipping inference.")
        return total, 0

    # Set up multiprocessing queues
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue(maxsize=len(pending) + len(GPU_IDS) + 10)
    result_queue = ctx.Queue()

    # Start workers
    workers = []
    active_gpus = []
    for gpu_id in GPU_IDS:
        p = ctx.Process(target=worker_process, args=(gpu_id, task_queue, result_queue), daemon=True)
        p.start()
        workers.append(p)

    # Wait for all workers to be ready
    ready_count = 0
    failed_gpus = []
    timeout_deadline = time.time() + 120
    while ready_count + len(failed_gpus) < len(GPU_IDS):
        if time.time() > timeout_deadline:
            logger.warning("Timeout waiting for GPU workers")
            break
        try:
            msg = result_queue.get(timeout=5)
            if msg[0] == "ready":
                active_gpus.append(msg[1])
                ready_count += 1
                logger.info(f"GPU {msg[1]} ready ({ready_count}/{len(GPU_IDS)})")
            elif msg[0] == "error":
                failed_gpus.append(msg[1])
                logger.error(f"GPU {msg[1]} failed: {msg[2]}")
        except Empty:
            pass

    if not active_gpus:
        logger.error("No GPU workers started. Aborting multi-GPU, trying single GPU.")
        for p in workers:
            p.terminate()
        # Fall back to single-GPU sequential
        return run_inference_single(samples, output_dir, dataset_name)

    logger.info(f"Running with {len(active_gpus)} GPUs: {active_gpus}")
    tracker.update(tracker.processed, f"inference on GPUs {active_gpus}")

    # Feed tasks to queue
    for item in pending:
        task_queue.put(item)

    # Send poison pills to stop workers
    for _ in workers:
        task_queue.put(None)

    # Collect results
    file_exists = os.path.exists(csv_path) and len(completed) > 0
    processed_count = len(completed)
    failed_count = 0
    collected = 0

    csv_f = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=["file_path", "probability", "predicted_label", "ground_truth"])
    if not file_exists:
        writer.writeheader()

    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(pending), desc=f"Evaluating {dataset_name}")

        deadline = time.time() + len(pending) * 30 + 3600  # generous timeout

        while collected < len(pending):
            if time.time() > deadline:
                logger.error("Timeout collecting results!")
                break

            # Check if workers died unexpectedly
            alive = sum(1 for p in workers if p.is_alive())

            try:
                msg = result_queue.get(timeout=30)
            except Empty:
                if alive == 0 and collected < len(pending):
                    logger.error(f"All workers died with {len(pending)-collected} tasks remaining")
                    break
                continue

            if msg[0] == "result":
                _, file_path, gt, prob, pred, error = msg
                if error:
                    failed_count += 1
                    tracker.add_error(f"{file_path}: {error}")
                    writer.writerow({
                        "file_path": file_path,
                        "probability": "",
                        "predicted_label": "",
                        "ground_truth": gt,
                    })
                else:
                    writer.writerow({
                        "file_path": file_path,
                        "probability": f"{prob:.6f}",
                        "predicted_label": pred,
                        "ground_truth": gt,
                    })

                collected += 1
                processed_count += 1
                pbar.update(1)

                if collected % 100 == 0:
                    tracker.update(processed_count)
                    csv_f.flush()

        pbar.close()
    finally:
        csv_f.flush()
        csv_f.close()

    # Clean up workers
    for p in workers:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    tracker.update(processed_count, "complete")
    return processed_count, failed_count


def run_inference_single(samples: list, output_dir: str, dataset_name: str) -> tuple:
    """Fallback: sequential inference on first available GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_IDS[0])

    cudnn_lib = "/data/awais/anaconda/envs/fsd/lib/python3.12/site-packages/nvidia/cudnn/lib"
    existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{cudnn_lib}:{existing_ld}" if existing_ld else cudnn_lib

    import torch
    from fsd import FSDDetector
    csv_path = os.path.join(output_dir, "predictions.csv")
    progress_file = os.path.join(output_dir, "progress.txt")

    completed = load_completed(csv_path)
    pending = [(p, l) for p, l in samples if p not in completed]
    total = len(samples)

    tracker = ProgressTracker(progress_file, dataset_name, total)
    tracker.processed = len(completed)

    logger.info(f"Loading FSD on GPU {GPU_IDS[0]}...")
    detector = FSDDetector.load(device="cuda")
    tracker.update(len(completed), "single-GPU inference")

    file_exists = os.path.exists(csv_path) and len(completed) > 0
    failed = 0
    processed_count = len(completed)

    from tqdm import tqdm
    with open(csv_path, "a", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=["file_path", "probability", "predicted_label", "ground_truth"])
        if not file_exists:
            writer.writeheader()

        for fp, gt in tqdm(pending, desc=f"Evaluating {dataset_name}"):
            try:
                result = detector.score(fp)
                z = result.z_score
                prob = 1.0 / (1.0 + (2.718281828 ** (z + 2.0)))
                pred = 1 if result.is_fake else 0
                writer.writerow({"file_path": fp, "probability": f"{prob:.6f}",
                                  "predicted_label": pred, "ground_truth": gt})
            except Exception as e:
                failed += 1
                tracker.add_error(f"{fp}: {e}")
                writer.writerow({"file_path": fp, "probability": "", "predicted_label": "", "ground_truth": gt})

            processed_count += 1
            if processed_count % 200 == 0:
                tracker.update(processed_count)
                csv_f.flush()

    tracker.update(processed_count, "complete")
    return processed_count, failed


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(csv_path: str) -> dict:
    from sklearn.metrics import roc_auc_score, accuracy_score

    rows = []
    skipped = 0
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if row["probability"] and row["predicted_label"] and row["ground_truth"]:
                    rows.append({
                        "prob": float(row["probability"]),
                        "pred": int(row["predicted_label"]),
                        "gt": int(row["ground_truth"]),
                    })
                else:
                    skipped += 1
            except (ValueError, KeyError):
                skipped += 1

    if not rows:
        return {"error": "No valid predictions", "skipped": skipped}

    probs = [r["prob"] for r in rows]
    preds = [r["pred"] for r in rows]
    gts = [r["gt"] for r in rows]

    acc = accuracy_score(gts, preds)
    try:
        auc = roc_auc_score(gts, probs)
    except Exception:
        auc = float("nan")

    real_rows = [r for r in rows if r["gt"] == 0]
    fake_rows = [r for r in rows if r["gt"] == 1]
    real_acc = sum(1 for r in real_rows if r["pred"] == 0) / max(len(real_rows), 1)
    fake_acc = sum(1 for r in fake_rows if r["pred"] == 1) / max(len(fake_rows), 1)

    return {
        "total": len(rows), "skipped": skipped,
        "real_count": len(real_rows), "fake_count": len(fake_rows),
        "accuracy": acc, "auc": auc,
        "real_accuracy": real_acc, "fake_accuracy": fake_acc,
    }


# ─── Summary ─────────────────────────────────────────────────────────────────

def write_result_summary(output_dir, dataset_name, dataset_path, samples, metrics, runtime, failed, mode):
    summary_path = os.path.join(output_dir, "result_summary.txt")
    real_count = sum(1 for _, l in samples if l == 0)
    fake_count = sum(1 for _, l in samples if l == 1)

    with open(summary_path, "w") as f:
        f.write(f"{'='*60}\nFSD Evaluation Summary: {dataset_name}\n{'='*60}\n\n")
        f.write(f"Dataset Information\n")
        f.write(f"  Dataset path:   {dataset_path}\n")
        f.write(f"  Total files:    {len(samples)}\n")
        f.write(f"  Real images:    {real_count} ({100*real_count/max(len(samples),1):.1f}%)\n")
        f.write(f"  Fake images:    {fake_count} ({100*fake_count/max(len(samples),1):.1f}%)\n\n")
        f.write(f"Preprocessing\n")
        f.write(f"  Formats:         jpg, jpeg, png, bmp, tiff, webp, heif, heic\n")
        f.write(f"  Grayscale:       Yes (FSD uses grayscale residuals)\n")
        f.write(f"  Resize:          Per model config (max_size from fsd config.json)\n\n")
        f.write(f"Model & Inference\n")
        f.write(f"  Model:           FSD (Forensic Self-Descriptions) CVPR 2025\n")
        f.write(f"  Weights:         Auto-downloaded from GitHub releases (~/.cache/fsd/)\n")
        f.write(f"  Inference mode:  {mode}\n")
        f.write(f"  GPU IDs:         {GPU_IDS}\n")
        f.write(f"  Threshold:       -2.0 (z-score; more negative = stricter)\n\n")
        f.write(f"Evaluation Results\n")
        if "error" in metrics:
            f.write(f"  ERROR: {metrics['error']}\n")
        else:
            f.write(f"  Accuracy:        {metrics['accuracy']:.4f} ({100*metrics['accuracy']:.2f}%)\n")
            f.write(f"  AUC:             {metrics['auc']:.4f}\n")
            f.write(f"  Real_accuracy:   {metrics['real_accuracy']:.4f} ({100*metrics['real_accuracy']:.2f}%)\n")
            f.write(f"  Fake_accuracy:   {metrics['fake_accuracy']:.4f} ({100*metrics['fake_accuracy']:.2f}%)\n")
            f.write(f"  Evaluated files: {metrics['total']}\n")
            f.write(f"  Skipped/failed:  {metrics.get('skipped',0) + failed}\n\n")
        f.write(f"Runtime\n")
        f.write(f"  Total:           {timedelta(seconds=int(runtime))}\n")
        f.write(f"  Throughput:      {len(samples)/max(runtime,1):.2f} img/sec\n\n")
        f.write(f"Output\n")
        f.write(f"  predictions.csv: {os.path.join(output_dir, 'predictions.csv')}\n")
        f.write(f"  progress.txt:    {os.path.join(output_dir, 'progress.txt')}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    logger.info(f"Summary: {summary_path}")
    return summary_path


# ─── Main ─────────────────────────────────────────────────────────────────────

def evaluate_dataset(dataset_name: str, config: dict) -> dict:
    dataset_path = config["path"]
    output_dir = os.path.join(RESULTS_BASE, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"\n{'='*60}\nEvaluating: {dataset_name}\nPath: {dataset_path}\n{'='*60}")
    start = time.time()

    samples = find_images_with_labels(dataset_path)
    if not samples:
        logger.error(f"No images found in {dataset_path}")
        return {"status": "failed", "error": "No images found"}

    real_c = sum(1 for _, l in samples if l == 0)
    fake_c = sum(1 for _, l in samples if l == 1)
    logger.info(f"{len(samples)} images: {real_c} real, {fake_c} fake")

    try:
        processed, failed = run_inference(samples, output_dir, dataset_name)
        mode = f"Multi-GPU {GPU_IDS}"
    except Exception as e:
        logger.error(f"Multi-GPU inference failed: {e}\n{traceback.format_exc()}")
        processed, failed = run_inference_single(samples, output_dir, dataset_name)
        mode = f"Single-GPU (GPU {GPU_IDS[0]})"

    runtime = time.time() - start
    logger.info(f"Done: {processed} processed, {failed} failed in {timedelta(seconds=int(runtime))}")

    csv_path = os.path.join(output_dir, "predictions.csv")
    metrics = compute_metrics(csv_path)
    logger.info(f"Metrics: {metrics}")

    write_result_summary(output_dir, dataset_name, dataset_path, samples, metrics, runtime, failed, mode)

    # Validate CSV
    with open(csv_path) as f:
        rows = sum(1 for _ in f) - 1
    if rows != len(samples):
        logger.warning(f"CSV rows ({rows}) != samples ({len(samples)})")
    else:
        logger.info(f"CSV validated: {rows} rows match {len(samples)} samples")

    return {
        "status": "success", "dataset": dataset_name,
        "total": len(samples), "processed": processed, "failed": failed,
        "metrics": metrics, "runtime": runtime, "output_dir": output_dir,
    }


def main():
    logger.info("=" * 70)
    logger.info("FSD Dataset Evaluation Pipeline")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"GPUs: {GPU_IDS}")
    logger.info(f"Results dir: {RESULTS_BASE}")
    logger.info("=" * 70)

    cmd_history = os.path.join(LOGS_DIR, "command_history.txt")
    with open(cmd_history, "a") as f:
        f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] python evaluate_datasets.py\n")
        f.write(f"  GPU_IDS={GPU_IDS}\n")
        f.write(f"  Datasets: {list(DATASETS.keys())}\n")

    all_results = {}
    pipeline_start = time.time()

    for dataset_name, config in DATASETS.items():
        logger.info(f"\n>>> {dataset_name} <<<")
        try:
            result = evaluate_dataset(dataset_name, config)
        except Exception as e:
            logger.error(f"FATAL {dataset_name}: {e}\n{traceback.format_exc()}")
            result = {"status": "failed", "error": str(e)}
        all_results[dataset_name] = result

    total_time = time.time() - pipeline_start
    report_path = os.path.join(RESULTS_BASE, "final_report.txt")

    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("FINAL EVALUATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total pipeline runtime: {timedelta(seconds=int(total_time))}\n")
        f.write("=" * 70 + "\n\n")

        for name, r in all_results.items():
            f.write(f"Dataset: {name}\n")
            f.write(f"  Status:   {r['status']}\n")
            if r["status"] == "success":
                m = r["metrics"]
                f.write(f"  Files:    {r['total']} total, {r['processed']} processed, {r['failed']} failed\n")
                if "error" not in m:
                    f.write(f"  Accuracy: {m['accuracy']:.4f}\n")
                    f.write(f"  AUC:      {m['auc']:.4f}\n")
                    f.write(f"  Real_acc: {m['real_accuracy']:.4f}\n")
                    f.write(f"  Fake_acc: {m['fake_accuracy']:.4f}\n")
                f.write(f"  Runtime:  {timedelta(seconds=int(r['runtime']))}\n")
                f.write(f"  Output:   {r['output_dir']}\n")
            else:
                f.write(f"  Error:    {r.get('error','unknown')}\n")
            f.write("\n")

    logger.info("=" * 70)
    with open(report_path) as f:
        logger.info(f.read())

    print(f"\nFinal report: {report_path}")
    print(f"All results:  {RESULTS_BASE}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
