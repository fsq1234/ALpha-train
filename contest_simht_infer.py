import argparse
import logging
import os
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Dict

import numpy as np
import torch

from contest_heavyrain_baseline import detect_heavyrain_regions, output_path, write_result
from datasets.zwmoc_reader import find_product_files, read_mosaic_bin, resize_for_model
from models.SimHT import get_model


ALGORITHM_NAME = "SimHT-CINRAD-NMProd-Heavyrain"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SimHT competition inference for ZW_MOC heavy-rain detection")
    parser.add_argument("--input_dir", default="/input/data", help="Competition input data directory")
    parser.add_argument("--output_dir", default="/output", help="Competition output directory")
    parser.add_argument("--log_dir", default="/log", help="Competition log directory")
    parser.add_argument("--ckpt", default="resources/simht_zwmoc.pt", help="Trained SimHT checkpoint")
    parser.add_argument("--product", default="QREF", choices=["QREF", "CREF"], help="ZW_MOC product used by SimHT")
    parser.add_argument("--img_size", type=int, default=128, help="Model input/output size")
    parser.add_argument("--frames_in", type=int, default=5, help="Input frame count")
    parser.add_argument("--frames_out", type=int, default=20, help="Forecast frame count")
    parser.add_argument("--lead_minutes", type=int, default=6, help="Minutes between output frames")
    parser.add_argument("--max_dbz", type=float, default=80.0, help="Normalization cap used in training")
    parser.add_argument("--threshold_dbz", type=float, default=40.0, help="Reflectivity threshold for regions")
    parser.add_argument("--min_area_pixels", type=int, default=20, help="Minimum region area on model grid")
    parser.add_argument("--max_regions", type=int, default=50, help="Maximum regions written per file")
    parser.add_argument("--max_polygon_points", type=int, default=80, help="Maximum points per polygon")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--allow_untrained",
        action="store_true",
        help="Run with random weights for wiring tests only; do not use for submission",
    )
    return parser


def clean_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    return cleaned


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device, allow_untrained: bool) -> None:
    if not os.path.exists(ckpt_path):
        if allow_untrained:
            logging.warning("Checkpoint %s not found; running with random weights", ckpt_path)
            return
        raise FileNotFoundError(f"SimHT checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint
    model.load_state_dict(clean_state_dict(state), strict=True)


def load_input_tensor(paths, img_size: int, max_dbz: float, device: torch.device):
    grids = [read_mosaic_bin(path) for path in paths]
    frames = [resize_for_model(grid.data, img_size, max_dbz=max_dbz) for grid in grids]
    array = np.stack(frames, axis=0)[:, None, :, :]
    tensor = torch.from_numpy(array).unsqueeze(0).to(device=device, dtype=torch.float32)
    return tensor, grids[-1]


def grid_for_prediction(reference_grid, prediction_dbz: np.ndarray):
    height, width = prediction_dbz.shape
    return replace(
        reference_grid,
        data=prediction_dbz.astype(np.float32),
        dx=(reference_grid.edge_e - reference_grid.edge_w) / float(width),
        dy=(reference_grid.edge_n - reference_grid.edge_s) / float(height),
    )


def main() -> None:
    args = create_parser().parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.log_dir, f"{ALGORITHM_NAME}-{datetime.utcnow():%Y%m%d%H%M%S}.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    device = torch.device(args.device)
    paths = find_product_files(args.input_dir, args.product)
    if len(paths) < args.frames_in:
        raise ValueError(f"Need at least {args.frames_in} {args.product} files, got {len(paths)}")

    input_paths = paths[-args.frames_in :]
    logging.info("Using input files: %s", input_paths)

    model = get_model(
        in_shape=(1, args.img_size, args.img_size),
        T_in=args.frames_in,
        T_out=args.frames_out,
    ).to(device)
    load_checkpoint(model, args.ckpt, device, args.allow_untrained)
    model.eval()

    input_tensor, reference_grid = load_input_tensor(input_paths, args.img_size, args.max_dbz, device)
    with torch.no_grad():
        prediction, _ = model.predict(input_tensor, compute_loss=False)

    prediction = prediction.squeeze(0).squeeze(1).detach().cpu().numpy()
    prediction_dbz = np.clip(prediction, 0.0, 1.0) * args.max_dbz

    for frame_index, frame in enumerate(prediction_dbz):
        valid_time = reference_grid.obs_time + timedelta(minutes=args.lead_minutes * (frame_index + 1))
        date_time = valid_time.strftime("%Y%m%d%H%M%S")
        pred_grid = grid_for_prediction(reference_grid, frame)
        features = detect_heavyrain_regions(
            pred_grid,
            threshold_dbz=args.threshold_dbz,
            min_area_pixels=args.min_area_pixels,
            max_regions=args.max_regions,
            max_polygon_points=args.max_polygon_points,
        )
        result_path = output_path(args.output_dir, date_time)
        write_result(result_path, date_time, features)
        logging.info("Wrote %s with %d regions", result_path, len(features))
        print(f"SimHT forecast {date_time} -> {result_path} ({len(features)} regions)")


if __name__ == "__main__":
    main()
