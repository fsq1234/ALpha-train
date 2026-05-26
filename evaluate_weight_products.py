import argparse
import logging
import os
import time
from dataclasses import replace
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch

from contest_heavyrain_baseline import detect_heavyrain_regions, output_path, write_result
from datasets.zwmoc_reader import group_product_files_by_time, read_mosaic_bin, resize_for_model
from utils.contest_metrics import average_file_seconds, csi, csi_skill_score, efficiency_skill_score, lead_skill_score


ALGORITHM_NAME = "SimHT-CINRAD-NMProd-Heavyrain"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SimHT heavy-rain recognition and log objective metrics")
    parser.add_argument("--input_dir", default="/input/data", help="Directory containing ZW_MOC data")
    parser.add_argument("--output_dir", default="/output/result", help="Directory for product JSON outputs")
    parser.add_argument("--log_dir", default="/log", help="Log directory")
    parser.add_argument("--ckpt", default="weight/ckpt-3068.pt", help="SimHT checkpoint path")
    parser.add_argument("--products", nargs="+", default=["QREF"], choices=["QREF", "CREF", "CAP"])
    parser.add_argument(
        "--product_output_subdirs",
        action="store_true",
        help="Write each product under output_dir/product for local multi-product debugging",
    )
    parser.add_argument("--cap_levels", nargs="*", type=int, default=None, help="Optional CAP layer indices, e.g. 0 1 2 3 4 5")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--frames_in", type=int, default=5)
    parser.add_argument("--frames_out", type=int, default=6)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_dbz", type=float, default=80.0)
    parser.add_argument("--threshold_dbz", type=float, default=40.0)
    parser.add_argument("--min_area_pixels", type=int, default=20)
    parser.add_argument("--max_regions", type=int, default=50)
    parser.add_argument("--max_polygon_points", type=int, default=80)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--baseline_csi", type=float, default=None, help="Reference SWAN/ROSE CSI for SCSI")
    parser.add_argument("--baseline_avg_seconds", type=float, default=None, help="Reference avg seconds/file for ESS")
    parser.add_argument("--baseline_lead_score", type=float, default=None, help="Reference lead score for LSS")
    parser.add_argument("--candidate_lead_score", type=float, default=None, help="Candidate lead score for LSS")
    return parser


def clean_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    return cleaned


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint
    model.load_state_dict(clean_state_dict(state), strict=True)


def cap_level(path: str) -> int:
    return int(os.path.splitext(os.path.basename(path))[0].split("_")[-1])


def read_product_grid(paths: List[str], product: str, cap_levels: Optional[List[int]]):
    if product != "CAP":
        return read_mosaic_bin(paths[0])

    selected_paths = paths
    if cap_levels is not None:
        selected_paths = [path for path in paths if cap_level(path) in cap_levels]
    if not selected_paths:
        raise ValueError("No CAP files matched cap_levels")

    base_grid = read_mosaic_bin(selected_paths[0])
    merged = base_grid.data
    for path in selected_paths[1:]:
        merged = np.maximum(merged, read_mosaic_bin(path).data)
    return replace(base_grid, data=merged, product="CAP")


def load_tensor_for_times(grouped, times: List[str], product: str, cap_levels, img_size: int, max_dbz: float, device):
    grids = [read_product_grid(grouped[time_key], product, cap_levels) for time_key in times]
    frames = [resize_for_model(grid.data, img_size, max_dbz=max_dbz) for grid in grids]
    array = np.stack(frames, axis=0)[:, None, :, :]
    tensor = torch.from_numpy(array).unsqueeze(0).to(device=device, dtype=torch.float32)
    return tensor, grids


def grid_for_prediction(reference_grid, prediction_dbz: np.ndarray):
    height, width = prediction_dbz.shape
    return replace(
        reference_grid,
        data=prediction_dbz.astype(np.float32),
        dx=(reference_grid.edge_e - reference_grid.edge_w) / float(width),
        dy=(reference_grid.edge_n - reference_grid.edge_s) / float(height),
    )


def confusion_counts(pred_dbz: np.ndarray, target_dbz: np.ndarray, threshold_dbz: float):
    pred_mask = pred_dbz >= threshold_dbz
    target_mask = target_dbz >= threshold_dbz
    hits = int(np.logical_and(pred_mask, target_mask).sum())
    false_alarms = int(np.logical_and(pred_mask, ~target_mask).sum())
    misses = int(np.logical_and(~pred_mask, target_mask).sum())
    return hits, false_alarms, misses


def evaluate_product(args, product: str, model: torch.nn.Module, device: torch.device):
    grouped = group_product_files_by_time(args.input_dir, product)
    times = sorted(grouped.keys())
    window_len = args.frames_in + args.frames_out
    if len(times) < window_len:
        raise ValueError(f"{product}: need at least {window_len} times, got {len(times)}")

    total_hits = total_false_alarms = total_misses = 0
    processed_files = 0
    json_count = 0
    start_time = time.perf_counter()

    for start in range(0, len(times) - window_len + 1, args.stride):
        input_times = times[start : start + args.frames_in]
        target_times = times[start + args.frames_in : start + window_len]

        input_tensor, _ = load_tensor_for_times(
            grouped,
            input_times,
            product,
            args.cap_levels,
            args.img_size,
            args.max_dbz,
            device,
        )
        _, target_grids = load_tensor_for_times(
            grouped,
            target_times,
            product,
            args.cap_levels,
            args.img_size,
            args.max_dbz,
            device,
        )

        with torch.no_grad():
            prediction, _ = model.predict(input_tensor, compute_loss=False)
        pred_norm = prediction.squeeze(0).squeeze(1).detach().cpu().numpy()
        pred_dbz = np.clip(pred_norm, 0.0, 1.0) * args.max_dbz

        for frame_index, target_grid in enumerate(target_grids):
            target_norm = resize_for_model(target_grid.data, args.img_size, max_dbz=args.max_dbz)
            target_dbz = target_norm * args.max_dbz
            hits, false_alarms, misses = confusion_counts(pred_dbz[frame_index], target_dbz, args.threshold_dbz)
            total_hits += hits
            total_false_alarms += false_alarms
            total_misses += misses

            pred_grid = grid_for_prediction(target_grid, pred_dbz[frame_index])
            features = detect_heavyrain_regions(
                pred_grid,
                threshold_dbz=args.threshold_dbz,
                min_area_pixels=args.min_area_pixels,
                max_regions=args.max_regions,
                max_polygon_points=args.max_polygon_points,
            )
            date_time = target_grid.obs_time.strftime("%Y%m%d%H%M%S")
            product_output_dir = os.path.join(args.output_dir, product) if args.product_output_subdirs else args.output_dir
            result_path = output_path(product_output_dir, date_time)
            write_result(result_path, date_time, features)
            logging.info("%s wrote %s with %d regions", product, result_path, len(features))
            print(f"{product} {date_time} -> {result_path} ({len(features)} regions)")
            json_count += 1

        processed_files += len(input_times) + len(target_times)

    total_seconds = time.perf_counter() - start_time
    product_csi = csi(total_hits, total_false_alarms, total_misses)
    avg_seconds = average_file_seconds(total_seconds, processed_files)
    scsi = csi_skill_score(product_csi, args.baseline_csi) if args.baseline_csi is not None else None
    ess = (
        efficiency_skill_score(avg_seconds, args.baseline_avg_seconds)
        if args.baseline_avg_seconds is not None
        else None
    )
    lss = (
        lead_skill_score(args.candidate_lead_score, args.baseline_lead_score)
        if args.candidate_lead_score is not None and args.baseline_lead_score is not None
        else None
    )
    metrics = {
        "product": product,
        "hits": total_hits,
        "false_alarms": total_false_alarms,
        "misses": total_misses,
        "csi": product_csi,
        "scsi": scsi,
        "lss": lss,
        "ess": ess,
        "total_seconds": total_seconds,
        "processed_files": processed_files,
        "avg_seconds_per_file": avg_seconds,
        "json_count": json_count,
    }
    logging.info(
        "METRICS product=%s hits=%d false_alarms=%d misses=%d CSI=%.6f SCSI=%s LSS=%s ESS=%s total_seconds=%.3f processed_files=%d avg_seconds_per_file=%.6f json_count=%d",
        product,
        total_hits,
        total_false_alarms,
        total_misses,
        product_csi,
        "N/A" if scsi is None else f"{scsi:.6f}",
        "N/A" if lss is None else f"{lss:.6f}",
        "N/A" if ess is None else f"{ess:.6f}",
        total_seconds,
        processed_files,
        avg_seconds,
        json_count,
    )
    print(
        f"{product} metrics: hits={total_hits} false_alarms={total_false_alarms} "
        f"misses={total_misses} CSI={product_csi:.6f} "
        f"SCSI={'N/A' if scsi is None else f'{scsi:.6f}'} "
        f"LSS={'N/A' if lss is None else f'{lss:.6f}'} "
        f"ESS={'N/A' if ess is None else f'{ess:.6f}'} "
        f"avg_seconds/file={avg_seconds:.6f}"
    )
    return metrics


def main() -> None:
    args = create_parser().parse_args()
    if len(args.products) > 1 and not args.product_output_subdirs:
        raise ValueError(
            "Multiple products would write the same filenames under one competition output directory. "
            "Use one final product, or add --product_output_subdirs for local debugging."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"weight-eval-{datetime.utcnow():%Y%m%d%H%M%S}.log")
    logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info(
        "Algorithm=%s input_dir=%s output_dir=%s log_dir=%s ckpt=%s products=%s product_output_subdirs=%s",
        ALGORITHM_NAME,
        args.input_dir,
        args.output_dir,
        args.log_dir,
        args.ckpt,
        ",".join(args.products),
        args.product_output_subdirs,
    )

    device = torch.device(args.device)
    from models.SimHT import get_model

    model = get_model(
        in_shape=(1, args.img_size, args.img_size),
        T_in=args.frames_in,
        T_out=args.frames_out,
    ).to(device)
    load_checkpoint(model, args.ckpt, device)
    model.eval()

    all_metrics = []
    for product in args.products:
        all_metrics.append(evaluate_product(args, product, model, device))
    logging.info("SUMMARY %s", all_metrics)
    print(f"log written to {log_path}")


if __name__ == "__main__":
    main()
