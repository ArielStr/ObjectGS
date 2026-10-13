
import argparse
import json
import re
from pathlib import Path

def summarize(output):
    log_path = output / "logs/training.log"
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    log = log_path.read_text(errors="replace")
    log = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", log)
    log = log.replace("\r", "\n")

    pattern = re.compile(
        r"\[MASKLOSS\]\s*"
        r"iter=(\d+)\s+"
        r"Obj=([\d.eE+-]+)\s+"
        r"Outside=([\d.eE+-]+)\s+"
        r"Total=([\d.eE+-]+)\s+"
        r"Mask=([\d.eE+-]+)\s+"
        r"G=(\d+)"
    )

    matches = list(pattern.finditer(log))
    if not matches:
        raise ValueError("No MASKLOSS records found")

    last = matches[-1]

    metrics = {
        "iterations": int(last.group(1)),
        "object_loss": float(last.group(2)),
        "outside_loss": float(last.group(3)),
        "total_mask_loss": float(last.group(4)),
        "mask_fraction": float(last.group(5)),
        "gaussian_count": int(last.group(6)),
    }

    eval_pattern = re.compile(
        r"\[ITER\s+(\d+)\]\s+Evaluating train:\s+"
        r"L1\s+([\d.eE+-]+)\s+"
        r"PSNR\s+([\d.eE+-]+)"
    )

    evaluations = list(eval_pattern.finditer(log))
    if evaluations:
        last_eval = evaluations[-1]
        metrics["evaluation_iteration"] = int(last_eval.group(1))
        metrics["evaluation_l1"] = float(last_eval.group(2))
        metrics["evaluation_psnr"] = float(last_eval.group(3))

    # Match only the primary tqdm Loss, not Depth Loss
    progress_pattern = re.compile(
        r"(?<!Depth )\bLoss=([\d.eE+-]+)"
    )

    progress = progress_pattern.findall(log)
    if progress:
        metrics["progress_loss"] = float(progress[-1])

    elapsed = re.findall(
        r"Training progress:\s*100%.*?\[(\d+:\d+)(?:<|,)",
        log
    )
    if elapsed:
        metrics["training_progress_elapsed"] = elapsed[-1]

    metrics["training_complete"] = "Training complete." in log

    destination = output / "training_metrics.json"
    destination.write_text(
        json.dumps(metrics, indent=2) + "\n"
    )

    print("===== OBJECTGS TRAINING SUMMARY =====")
    for key, value in metrics.items():
        print(f"{key:28s}: {value}")

    print(f"\nSaved: {destination}")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.output)
