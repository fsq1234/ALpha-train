import argparse
import os
import sys
from typing import Dict, List, Optional

import h5py
import numpy as np
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from datasets.zwmoc_reader import group_product_files_by_time, read_mosaic_bin, resize_for_model


def to_uint8(frame: np.ndarray) -> np.ndarray:
    return np.clip(frame * 255.0, 0, 255).astype(np.uint8)


def cap_level(path: str) -> int:
    return int(os.path.splitext(os.path.basename(path))[0].split("_")[-1])


def aggregate_cap(paths: List[str], cap_levels: Optional[List[int]]) -> np.ndarray:
    if cap_levels is not None:
        paths = [path for path in paths if cap_level(path) in cap_levels]
    if not paths:
        raise ValueError("No CAP files matched the requested cap_levels")
    merged = None
    for path in paths:
        data = read_mosaic_bin(path).data
        merged = data if merged is None else np.maximum(merged, data)
    return merged


def load_channel(paths: List[str], product: str, cap_levels: Optional[List[int]]) -> np.ndarray:
    if product == "CAP":
        return aggregate_cap(paths, cap_levels)
    return read_mosaic_bin(paths[0]).data


def build_frame(
    time_key: str,
    grouped: Dict[str, Dict[str, List[str]]],
    products: List[str],
    img_size: int,
    max_dbz: float,
    cap_levels: Optional[List[int]],
):
    channels = []
    for product in products:
        frame = load_channel(grouped[product][time_key], product, cap_levels)
        channels.append(to_uint8(resize_for_model(frame, img_size, max_dbz=max_dbz)))
    if len(channels) == 1:
        return channels[0]
    return np.stack(channels, axis=0)


def precompute_frames(
    times: List[str],
    grouped: Dict[str, Dict[str, List[str]]],
    products: List[str],
    img_size: int,
    max_dbz: float,
    cap_levels: Optional[List[int]],
) -> Dict[str, np.ndarray]:
    frames = {}
    for time_key in tqdm(times, desc="precompute frames"):
        frames[time_key] = build_frame(time_key, grouped, products, img_size, max_dbz, cap_levels)
    return frames


def write_split(
    group,
    sequences: List[List[str]],
    frames_by_time: Dict[str, np.ndarray],
    compression: str,
) -> None:
    for index, sequence_times in enumerate(tqdm(sequences, desc=f"write {group.name}")):
        frames = [frames_by_time[time_key] for time_key in sequence_times]
        group.create_dataset(str(index), data=np.stack(frames, axis=0), dtype="uint8", compression=compression)
    group.create_dataset("all_len", data=len(sequences))


def common_times(grouped: Dict[str, Dict[str, List[str]]], products: List[str]) -> List[str]:
    times = set(grouped[products[0]].keys())
    for product in products[1:]:
        times &= set(grouped[product].keys())
    return sorted(times)


def iter_time_sequences(times: List[str], seq_len: int):
    for start in range(0, len(times) - seq_len + 1):
        yield times[start : start + seq_len]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build SimHT h5 training data from ZW_MOC mosaic products")
    parser.add_argument("--input_dir", default="data", help="Directory containing ZW_MOC/YYYY/YYYYMMDD/product/*.bin")
    parser.add_argument("--output", default="data/zwmoc_qref.h5", help="Output h5 path")
    parser.add_argument("--product", default="QREF", choices=["QREF", "CREF", "CAP"], help="Single product shortcut")
    parser.add_argument(
        "--products",
        nargs="+",
        default=None,
        choices=["QREF", "CREF", "CAP"],
        help="Products fused as channels, e.g. --products QREF CREF CAP",
    )
    parser.add_argument("--seq_len", type=int, default=10, help="Total sequence length, e.g. frames_in + frames_out")
    parser.add_argument("--img_size", type=int, default=128, help="Stored frame size")
    parser.add_argument("--max_dbz", type=float, default=80.0, help="Reflectivity cap used for 0-1 normalization")
    parser.add_argument("--stride", type=int, default=1, help="Stride between sampled sequences")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--compression", default="lzf", choices=["lzf", "gzip", "none"])
    parser.add_argument(
        "--cap_levels",
        nargs="*",
        type=int,
        default=None,
        help="Optional CAP layer indices to fuse, e.g. --cap_levels 0 1 2 3 4 5",
    )
    return parser


def main() -> None:
    args = create_parser().parse_args()
    products = args.products if args.products else [args.product]
    grouped = {product: group_product_files_by_time(args.input_dir, product) for product in products}
    times = common_times(grouped, products)
    if len(times) < args.seq_len:
        raise ValueError(f"Need at least {args.seq_len} common times for {products}, got {len(times)}")

    sequences = list(iter_time_sequences(times, args.seq_len))[:: args.stride]
    split_index = max(1, int(len(sequences) * args.train_ratio))
    if split_index >= len(sequences):
        split_index = max(1, len(sequences) - 1)
    train_sequences = sequences[:split_index]
    test_sequences = sequences[split_index:]
    if not test_sequences:
        test_sequences = train_sequences[-1:]
    used_times = sorted({time_key for sequence in train_sequences + test_sequences for time_key in sequence})
    frames_by_time = precompute_frames(used_times, grouped, products, args.img_size, args.max_dbz, args.cap_levels)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    compression = None if args.compression == "none" else args.compression
    with h5py.File(args.output, "w") as h5_file:
        h5_file.attrs["product"] = ",".join(products)
        h5_file.attrs["channels"] = len(products)
        h5_file.attrs["seq_len"] = args.seq_len
        h5_file.attrs["img_size"] = args.img_size
        h5_file.attrs["max_dbz"] = args.max_dbz
        if args.cap_levels is not None:
            h5_file.attrs["cap_levels"] = ",".join(str(level) for level in args.cap_levels)
        write_split(h5_file.create_group("train"), train_sequences, frames_by_time, compression)
        write_split(h5_file.create_group("test"), test_sequences, frames_by_time, compression)

    print(f"Wrote {args.output}: train={len(train_sequences)}, test={len(test_sequences)}")


if __name__ == "__main__":
    main()
