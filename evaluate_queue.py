"""
FSD evaluation script for queued datasets: image_eval24 and ReWIND.
Uses GPUs 2, 3, 4 — does NOT touch the AIGI_TEST run on GPUs 5, 6, 7.
Resume-safe: re-reads existing CSV rows and skips already-processed files.
"""

import os
import sys
import csv
import time
import logging
import traceback
import subprocess
import multiprocessing as mp
from pathlib import Path
from datetime import datetime, timedelta
from queue import Empty

# ─── Configuration ────────────────────────────────────────────────────────────

DATASETS = {
    "image_eval24": {
        "path": "/mnt/h200_dataset/saad/datasets/images/image_eval24",
    },
    "ReWIND": {
        "path": "/mnt/h200_dataset/saad/datasets/images/ancestree/ReWIND",
    },
}

RESULTS_BASE  = "/data/awais/projects/FSD/Results"
LOGS_DIR      = "/data/awais/projects/FSD/logs"
GPU_IDS       = [2, 3, 4]          # separate from AIGI run on 5,6,7
CUDNN_LIB     = "/data/awais/anaconda/envs/fsd/lib/python3.12/site-packages/nvidia/cudnn/lib"
IMAGE_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".heif", ".heic"}
# Pre-resize images larger than this on either dimension before FRE
# Prevents OOM on 50MP+ images (im2col for 15x15 conv on 8640x5760 float64 ≈ 90 GB)
PRE_RESIZE_MAX = 2048

# ─── Logging ─────────────────────────────────────────────────────────────────

os.makedirs(LOGS_DIR, exist_ok=True)
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, f"queue_{_ts}.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
_err_handler = logging.FileHandler(os.path.join(LOGS_DIR, f"queue_errors_{_ts}.log"))
_err_handler.setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
logger.addHandler(_err_handler)


# ─── Dataset discovery ────────────────────────────────────────────────────────

def find_images(dataset_path: str) -> list:
    root = Path(dataset_path)
    samples = []
    for label_dir in root.rglob("0_real"):
        if label_dir.is_dir():
            for f in label_dir.rglob("*"):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                    samples.append((str(f), 0))
    for label_dir in root.rglob("1_fake"):
        if label_dir.is_dir():
            for f in label_dir.rglob("*"):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                    samples.append((str(f), 1))
    return samples


# ─── Resume support ───────────────────────────────────────────────────────────

def load_completed(csv_path: str) -> set:
    """Return set of file paths that already have a valid (non-empty) result.
    Files with empty probability (failed rows) are NOT counted as done so
    they will be retried on resume."""
    done = set()
    if not os.path.exists(csv_path):
        return done
    try:
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                fp   = row.get("file_path", "")
                prob = row.get("probability", "")
                if fp and prob:          # only count successfully scored rows
                    done.add(fp)
        logger.info(f"Resume: {len(done)} successfully scored files found in {csv_path}")
    except Exception as e:
        logger.warning(f"Could not read existing CSV: {e}")
    return done


# ─── Progress tracker ─────────────────────────────────────────────────────────

class ProgressTracker:
    def __init__(self, path, name, total):
        self.path = path
        self.name = name
        self.total = total
        self.processed = 0
        self.start = time.time()
        self.stage = "initializing"
        self.errors = []
        self._gpu_ts = 0
        self._gpu_cache = ""

    def update(self, n, stage=None):
        self.processed = n
        if stage:
            self.stage = stage
        self._write()

    def add_error(self, msg):
        self.errors.append(msg)

    def _gpu(self):
        if time.time() - self._gpu_ts < 30:
            return self._gpu_cache
        try:
            ids = ",".join(str(g) for g in GPU_IDS)
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader", f"--id={ids}"],
                capture_output=True, text=True, timeout=5)
            self._gpu_cache = r.stdout.strip()
        except Exception:
            self._gpu_cache = "unavailable"
        self._gpu_ts = time.time()
        return self._gpu_cache

    def _write(self):
        elapsed = time.time() - self.start
        rate = self.processed / elapsed if elapsed > 0 else 0
        rem = (self.total - self.processed) / rate if rate > 0 else 0
        eta = datetime.now() + timedelta(seconds=rem)
        pct = 100 * self.processed / self.total if self.total > 0 else 0
        try:
            with open(self.path, "w") as f:
                f.write(f"Dataset: {self.name}\n")
                f.write(f"Stage: {self.stage}\n")
                f.write(f"Progress: {self.processed}/{self.total} ({pct:.1f}%)\n")
                f.write(f"Elapsed: {timedelta(seconds=int(elapsed))}\n")
                f.write(f"Rate: {rate:.2f} img/sec\n")
                f.write(f"ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')} ({timedelta(seconds=int(rem))} remaining)\n")
                f.write(f"GPUs: {GPU_IDS}\n")
                f.write(f"GPU Usage:\n{self._gpu()}\n")
                if self.errors:
                    f.write(f"\nErrors ({len(self.errors)}):\n")
                    for e in self.errors[-10:]:
                        f.write(f"  {e}\n")
        except Exception:
            pass


# ─── Image loading with pre-resize ───────────────────────────────────────────

def load_image_safe(file_path: str):
    """Load image as grayscale PIL, pre-downsizing if larger than PRE_RESIZE_MAX
    on either dimension to avoid GPU OOM on very high-res images."""
    from PIL import Image as PILImage
    img = PILImage.open(file_path)
    if img.mode != "L":
        img = img.convert("L")
    w, h = img.size
    if w > PRE_RESIZE_MAX or h > PRE_RESIZE_MAX:
        scale = PRE_RESIZE_MAX / max(w, h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        img = img.resize((new_w, new_h), PILImage.LANCZOS)
    return img


# ─── Worker process ───────────────────────────────────────────────────────────

def worker_process(gpu_id: int, task_q: mp.Queue, result_q: mp.Queue):
    # Must set env vars before torch import so cuDNN is found
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{CUDNN_LIB}:{existing}" if existing else CUDNN_LIB
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    try:
        import torch
        from fsd import FSDDetector
        detector = FSDDetector.load(device="cuda")
        result_q.put(("ready", gpu_id, None))
    except Exception as e:
        result_q.put(("error", gpu_id, str(e)))
        return

    import torch

    while True:
        try:
            item = task_q.get(timeout=60)
        except Empty:
            break
        if item is None:
            break
        file_path, gt = item
        try:
            # Pre-resize large images to prevent GPU OOM
            img = load_image_safe(file_path)
            r   = detector.score(img)
            z   = r.z_score
            prob = 1.0 / (1.0 + (2.718281828 ** (z + 2.0)))
            pred = 1 if r.is_fake else 0
            result_q.put(("result", file_path, gt, prob, pred, None))
        except torch.cuda.OutOfMemoryError as e:
            # OOM: clear cache and retry on CPU
            torch.cuda.empty_cache()
            try:
                cpu_det = FSDDetector.load(device="cpu")
                img = load_image_safe(file_path)
                r   = cpu_det.score(img)
                z   = r.z_score
                prob = 1.0 / (1.0 + (2.718281828 ** (z + 2.0)))
                pred = 1 if r.is_fake else 0
                result_q.put(("result", file_path, gt, prob, pred, None))
                del cpu_det
                torch.cuda.empty_cache()
            except Exception as e2:
                result_q.put(("result", file_path, gt, None, None, f"OOM+CPU-fallback: {e2}"))
        except Exception as e:
            torch.cuda.empty_cache()
            result_q.put(("result", file_path, gt, None, None, str(e)))


# ─── Multi-GPU inference ──────────────────────────────────────────────────────

def run_inference(samples: list, output_dir: str, name: str) -> tuple:
    csv_path      = os.path.join(output_dir, "predictions.csv")
    progress_file = os.path.join(output_dir, "progress.txt")

    done    = load_completed(csv_path)
    pending = [(p, l) for p, l in samples if p not in done]
    total   = len(samples)

    tracker = ProgressTracker(progress_file, name, total)
    tracker.processed = len(done)
    tracker.update(len(done), "starting")

    logger.info(f"Total: {total} | Done: {len(done)} | Pending: {len(pending)}")
    if not pending:
        logger.info("All files already processed — skipping inference.")
        return total, 0

    ctx = mp.get_context("spawn")
    task_q   = ctx.Queue(maxsize=len(pending) + len(GPU_IDS) + 10)
    result_q = ctx.Queue()

    workers = []
    for gid in GPU_IDS:
        p = ctx.Process(target=worker_process, args=(gid, task_q, result_q), daemon=True)
        p.start()
        workers.append(p)

    # Wait for workers to be ready
    ready, failed = 0, []
    deadline = time.time() + 120
    while ready + len(failed) < len(GPU_IDS):
        if time.time() > deadline:
            logger.warning("Timeout waiting for workers")
            break
        try:
            msg = result_q.get(timeout=5)
            if msg[0] == "ready":
                ready += 1
                logger.info(f"GPU {msg[1]} ready ({ready}/{len(GPU_IDS)})")
            elif msg[0] == "error":
                failed.append(msg[1])
                logger.error(f"GPU {msg[1]} init failed: {msg[2]}")
        except Empty:
            pass

    if ready == 0:
        logger.error("No GPU workers available — falling back to single GPU")
        for p in workers:
            p.terminate()
        return run_inference_single(samples, output_dir, name)

    logger.info(f"Running on GPUs: {GPU_IDS[:ready]}")
    tracker.update(tracker.processed, f"inference on GPUs {GPU_IDS}")

    for item in pending:
        task_q.put(item)
    for _ in workers:
        task_q.put(None)

    file_exists = os.path.exists(csv_path) and len(done) > 0
    csv_f  = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=["file_path", "probability", "predicted_label", "ground_truth"])
    if not file_exists:
        writer.writeheader()

    processed_count = len(done)
    failed_count    = 0
    collected       = 0
    deadline_inf    = time.time() + len(pending) * 30 + 7200

    from tqdm import tqdm
    pbar = tqdm(total=len(pending), desc=f"Evaluating {name}", ncols=80)

    try:
        while collected < len(pending):
            if time.time() > deadline_inf:
                logger.error("Inference timeout!")
                break
            alive = sum(1 for p in workers if p.is_alive())
            try:
                msg = result_q.get(timeout=30)
            except Empty:
                if alive == 0 and collected < len(pending):
                    logger.error(f"All workers died with {len(pending)-collected} tasks left")
                    break
                continue

            if msg[0] == "result":
                _, fp, gt, prob, pred, err = msg
                if err:
                    failed_count += 1
                    tracker.add_error(f"{fp}: {err}")
                    writer.writerow({"file_path": fp, "probability": "", "predicted_label": "", "ground_truth": gt})
                else:
                    writer.writerow({"file_path": fp, "probability": f"{prob:.6f}",
                                     "predicted_label": pred, "ground_truth": gt})
                collected       += 1
                processed_count += 1
                pbar.update(1)
                if collected % 50 == 0:
                    tracker.update(processed_count)
                    csv_f.flush()
    finally:
        pbar.close()
        csv_f.flush()
        csv_f.close()

    for p in workers:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    tracker.update(processed_count, "complete")
    return processed_count, failed_count


def run_inference_single(samples: list, output_dir: str, name: str) -> tuple:
    """Single-GPU fallback."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_IDS[0])
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{CUDNN_LIB}:{existing}" if existing else CUDNN_LIB

    import torch
    from fsd import FSDDetector

    csv_path = os.path.join(output_dir, "predictions.csv")
    done     = load_completed(csv_path)
    pending  = [(p, l) for p, l in samples if p not in done]

    tracker = ProgressTracker(os.path.join(output_dir, "progress.txt"), name, len(samples))
    tracker.processed = len(done)

    detector = FSDDetector.load(device="cuda")
    tracker.update(len(done), f"single-GPU inference (GPU {GPU_IDS[0]})")

    file_exists = os.path.exists(csv_path) and len(done) > 0
    failed = 0
    count  = len(done)

    from tqdm import tqdm
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_path", "probability", "predicted_label", "ground_truth"])
        if not file_exists:
            writer.writeheader()
        for fp, gt in tqdm(pending, desc=f"Evaluating {name}", ncols=80):
            try:
                img  = load_image_safe(fp)
                r    = detector.score(img)
                prob = 1.0 / (1.0 + (2.718281828 ** (r.z_score + 2.0)))
                pred = 1 if r.is_fake else 0
                writer.writerow({"file_path": fp, "probability": f"{prob:.6f}",
                                 "predicted_label": pred, "ground_truth": gt})
            except Exception as e:
                failed += 1
                tracker.add_error(f"{fp}: {e}")
                writer.writerow({"file_path": fp, "probability": "", "predicted_label": "", "ground_truth": gt})
            count += 1
            if count % 100 == 0:
                tracker.update(count)
                f.flush()

    tracker.update(count, "complete")
    return count, failed


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(csv_path: str) -> dict:
    from sklearn.metrics import roc_auc_score, accuracy_score
    rows, skipped = [], 0
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                if row["probability"] and row["predicted_label"] and row["ground_truth"]:
                    rows.append({"prob": float(row["probability"]),
                                 "pred": int(row["predicted_label"]),
                                 "gt":   int(row["ground_truth"])})
                else:
                    skipped += 1
            except (ValueError, KeyError):
                skipped += 1
    if not rows:
        return {"error": "No valid predictions", "skipped": skipped}
    probs = [r["prob"] for r in rows]
    preds = [r["pred"] for r in rows]
    gts   = [r["gt"]   for r in rows]
    acc   = accuracy_score(gts, preds)
    try:
        auc = roc_auc_score(gts, probs)
    except Exception:
        auc = float("nan")
    real_rows = [r for r in rows if r["gt"] == 0]
    fake_rows = [r for r in rows if r["gt"] == 1]
    return {
        "total": len(rows), "skipped": skipped,
        "real_count": len(real_rows), "fake_count": len(fake_rows),
        "accuracy":      acc,
        "auc":           auc,
        "real_accuracy": sum(1 for r in real_rows if r["pred"] == 0) / max(len(real_rows), 1),
        "fake_accuracy": sum(1 for r in fake_rows if r["pred"] == 1) / max(len(fake_rows), 1),
    }


# ─── Summary ─────────────────────────────────────────────────────────────────

def write_summary(output_dir, name, path, samples, metrics, runtime, failed, mode):
    real_c = sum(1 for _, l in samples if l == 0)
    fake_c = sum(1 for _, l in samples if l == 1)
    summary = os.path.join(output_dir, "result_summary.txt")
    with open(summary, "w") as f:
        f.write(f"{'='*60}\nFSD Evaluation Summary: {name}\n{'='*60}\n\n")
        f.write(f"Dataset Information\n")
        f.write(f"  Dataset path:   {path}\n")
        f.write(f"  Total files:    {len(samples)}\n")
        f.write(f"  Real images:    {real_c} ({100*real_c/max(len(samples),1):.1f}%)\n")
        f.write(f"  Fake images:    {fake_c} ({100*fake_c/max(len(samples),1):.1f}%)\n\n")
        f.write(f"Preprocessing\n")
        f.write(f"  Formats:        jpg, jpeg, png, bmp, tiff, webp, heif, heic\n")
        f.write(f"  Grayscale:      Yes (FSD operates on grayscale residuals)\n")
        f.write(f"  Resize mode:    resize_and_crop to max_size=1024\n\n")
        f.write(f"Model & Inference\n")
        f.write(f"  Model:          FSD (Forensic Self-Descriptions) CVPR 2025\n")
        f.write(f"  Weights:        ~/.cache/fsd/v1.2.0/ (auto-downloaded)\n")
        f.write(f"  Inference mode: {mode}\n")
        f.write(f"  GPU IDs:        {GPU_IDS}\n")
        f.write(f"  Threshold:      -2.0 (z-score; z < -2.0 → FAKE)\n\n")
        f.write(f"Evaluation Results\n")
        if "error" in metrics:
            f.write(f"  ERROR: {metrics['error']}\n")
        else:
            f.write(f"  Accuracy:       {metrics['accuracy']:.4f}  ({100*metrics['accuracy']:.2f}%)\n")
            f.write(f"  AUC:            {metrics['auc']:.4f}\n")
            f.write(f"  Real_accuracy:  {metrics['real_accuracy']:.4f}  ({100*metrics['real_accuracy']:.2f}%)\n")
            f.write(f"  Fake_accuracy:  {metrics['fake_accuracy']:.4f}  ({100*metrics['fake_accuracy']:.2f}%)\n")
            f.write(f"  Evaluated:      {metrics['total']} files\n")
            f.write(f"  Skipped/failed: {metrics.get('skipped',0) + failed}\n\n")
        f.write(f"Runtime\n")
        f.write(f"  Total:          {timedelta(seconds=int(runtime))}\n")
        f.write(f"  Throughput:     {len(samples)/max(runtime,1):.2f} img/sec\n\n")
        f.write(f"Output files\n")
        f.write(f"  {os.path.join(output_dir, 'predictions.csv')}\n")
        f.write(f"  {os.path.join(output_dir, 'progress.txt')}\n")
        f.write(f"  {summary}\n")
        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    logger.info(f"Summary written: {summary}")


# ─── Per-dataset runner ───────────────────────────────────────────────────────

def evaluate_dataset(name: str, config: dict) -> dict:
    path       = config["path"]
    output_dir = os.path.join(RESULTS_BASE, name)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"\n{'='*60}\nEvaluating: {name}\nPath: {path}\n{'='*60}")
    t0 = time.time()

    samples = find_images(path)
    if not samples:
        logger.error(f"No images found in {path}")
        return {"status": "failed", "error": "No images found"}

    real_c = sum(1 for _, l in samples if l == 0)
    fake_c = sum(1 for _, l in samples if l == 1)
    logger.info(f"{len(samples)} images: {real_c} real, {fake_c} fake")

    try:
        processed, failed = run_inference(samples, output_dir, name)
        mode = f"Multi-GPU {GPU_IDS}"
    except Exception as e:
        logger.error(f"Multi-GPU failed: {e}\n{traceback.format_exc()}")
        processed, failed = run_inference_single(samples, output_dir, name)
        mode = f"Single-GPU (GPU {GPU_IDS[0]})"

    runtime = time.time() - t0
    logger.info(f"Done: {processed} processed, {failed} failed in {timedelta(seconds=int(runtime))}")

    csv_path = os.path.join(output_dir, "predictions.csv")
    metrics  = compute_metrics(csv_path)
    logger.info(f"Metrics: {metrics}")

    write_summary(output_dir, name, path, samples, metrics, runtime, failed, mode)

    # Validate row count
    with open(csv_path) as f:
        rows = sum(1 for _ in f) - 1
    if rows != len(samples):
        logger.warning(f"Row count mismatch: CSV={rows}, expected={len(samples)}")
    else:
        logger.info(f"CSV validated: {rows} rows")

    return {"status": "success", "dataset": name, "total": len(samples),
            "processed": processed, "failed": failed, "metrics": metrics,
            "runtime": runtime, "output_dir": output_dir}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("FSD Queue Evaluation  —  GPUs 2, 3, 4")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Datasets: {list(DATASETS.keys())}")
    logger.info("NOTE: AIGI_TEST is running separately on GPUs 5, 6, 7")
    logger.info("=" * 60)

    with open(os.path.join(LOGS_DIR, "command_history.txt"), "a") as f:
        f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] python evaluate_queue.py\n")
        f.write(f"  GPU_IDS={GPU_IDS}\n")
        f.write(f"  Datasets: {list(DATASETS.keys())}\n")

    all_results = {}
    t_start = time.time()

    for name, config in DATASETS.items():
        logger.info(f"\n>>> {name} <<<")
        try:
            result = evaluate_dataset(name, config)
        except Exception as e:
            logger.error(f"FATAL {name}: {e}\n{traceback.format_exc()}")
            result = {"status": "failed", "error": str(e)}
        all_results[name] = result

    total_t = time.time() - t_start
    report  = os.path.join(RESULTS_BASE, "queue_final_report.txt")

    with open(report, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("QUEUE EVALUATION FINAL REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total runtime: {timedelta(seconds=int(total_t))}\n")
        f.write("=" * 60 + "\n\n")
        for name, r in all_results.items():
            f.write(f"Dataset: {name}\n  Status: {r['status']}\n")
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

    logger.info("=" * 60)
    with open(report) as f:
        logger.info(f.read())
    print(f"\nFinal report: {report}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
