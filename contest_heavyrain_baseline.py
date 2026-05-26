import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict, List

import cv2
import numpy as np

from datasets.zwmoc_reader import find_product_files, parse_time_from_name, pixel_to_lonlat, read_mosaic_bin


ALGORITHM_NAME = "SimHT-CINRAD-NMProd-Heavyrain"


def dbz_to_rain_rate(dbz: float) -> float:
    """Convert reflectivity to a rough rain-rate estimate with Z=200R^1.6."""
    z = 10 ** (dbz / 10.0)
    return float((z / 200.0) ** (1.0 / 1.6))


def contour_to_feature(grid, contour: np.ndarray, max_points: int) -> Dict:
    epsilon = max(1.0, 0.002 * cv2.arcLength(contour, closed=True))
    approx = cv2.approxPolyDP(contour, epsilon, closed=True).reshape(-1, 2)

    if len(approx) > max_points:
        step = int(np.ceil(len(approx) / max_points))
        approx = approx[::step]

    polygon = [list(pixel_to_lonlat(grid, float(x), float(y))) for x, y in approx]
    if len(polygon) < 3:
        return {}
    if polygon[0] != polygon[-1]:
        polygon.append(polygon[0])

    moments = cv2.moments(contour)
    if moments["m00"]:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
    else:
        cx, cy = approx[:, 0].mean(), approx[:, 1].mean()
    lon, lat = pixel_to_lonlat(grid, cx, cy)

    mask = np.zeros(grid.data.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], contourIdx=-1, color=1, thickness=-1)
    max_dbz = float(grid.data[mask == 1].max())

    return {
        "lon": lon,
        "lat": lat,
        "polygon": polygon,
        "data": round(dbz_to_rain_rate(max_dbz), 2),
        "max_dbz": round(max_dbz, 2),
    }


def detect_heavyrain_regions(
    grid,
    threshold_dbz: float,
    min_area_pixels: int,
    max_regions: int,
    max_polygon_points: int,
) -> List[Dict]:
    mask = (grid.data >= threshold_dbz).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    features = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area_pixels:
            continue
        feature = contour_to_feature(grid, contour, max_polygon_points)
        if feature:
            features.append(feature)
        if len(features) >= max_regions:
            break
    return features


def output_path(output_dir: str, date_time: str) -> str:
    year = date_time[:4]
    ymd = date_time[:8]
    out_dir = os.path.join(output_dir, year, ymd, "heavyrain")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{date_time}_heavyrain.json")


def write_result(path: str, date_time: str, features: List[Dict]) -> None:
    payload = {
        "site_code": "",
        "date_time": date_time,
        "datas": features,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ZW_MOC QREF short-term heavy-rain baseline")
    parser.add_argument("--input_dir", default="/input/data", help="Competition input data directory")
    parser.add_argument("--output_dir", default="/output", help="Competition output directory")
    parser.add_argument("--log_dir", default="/log", help="Competition log directory")
    parser.add_argument("--product", default="QREF", choices=["QREF", "CREF"], help="ZW_MOC product to use")
    parser.add_argument("--threshold_dbz", type=float, default=45.0, help="Reflectivity threshold for regions")
    parser.add_argument("--min_area_pixels", type=int, default=100, help="Minimum connected-region area")
    parser.add_argument("--max_regions", type=int, default=50, help="Maximum regions written per file")
    parser.add_argument("--max_polygon_points", type=int, default=80, help="Maximum points per polygon")
    return parser


def main() -> None:
    args = create_parser().parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.log_dir, f"{ALGORITHM_NAME}-{datetime.utcnow():%Y%m%d%H%M%S}.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    paths = find_product_files(args.input_dir, args.product)
    logging.info("Found %d %s files under %s", len(paths), args.product, args.input_dir)
    if not paths:
        print(f"No {args.product} files found under {args.input_dir}")
        return

    for path in paths:
        try:
            grid = read_mosaic_bin(path)
            date_time = parse_time_from_name(path)
            features = detect_heavyrain_regions(
                grid,
                threshold_dbz=args.threshold_dbz,
                min_area_pixels=args.min_area_pixels,
                max_regions=args.max_regions,
                max_polygon_points=args.max_polygon_points,
            )
            result_path = output_path(args.output_dir, date_time)
            write_result(result_path, date_time, features)
            logging.info("Wrote %s with %d regions", result_path, len(features))
            print(f"{os.path.basename(path)} -> {result_path} ({len(features)} regions)")
        except Exception:
            logging.exception("Failed to process %s", path)
            raise


if __name__ == "__main__":
    main()
