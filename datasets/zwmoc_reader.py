import bz2
import glob
import os
import re
import struct
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


MOSAIC_HEADER_FORMAT = (
    "<4s4sihh8s64s"
    "iii"
    "hhhhhh"
    "iHHi"
    "iiiiiiiiii"
    "hhii"
    "hh8s8s60s"
)


@dataclass
class MosaicGrid:
    data: np.ndarray
    product: str
    obs_time: datetime
    edge_w: float
    edge_s: float
    edge_e: float
    edge_n: float
    dx: float
    dy: float
    scale: float
    path: str


def _decode_ascii(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("ascii", errors="ignore")


def read_mosaic_bin(path: str, fill_value: float = 0.0) -> MosaicGrid:
    """Read a CINRAD V3.0 national mosaic product.

    The files in ZW_MOC are little-endian MOC files with a 256-byte header and
    an optional bzip2-compressed int16 data block.
    """
    with open(path, "rb") as handle:
        content = handle.read()

    header_size = struct.calcsize(MOSAIC_HEADER_FORMAT)
    if len(content) < 256 or header_size != 256:
        raise ValueError(f"Invalid mosaic header in {path}")

    values = struct.unpack(MOSAIC_HEADER_FORMAT, content[:256])
    (
        label,
        _version,
        _file_bytes,
        _mosaic_id,
        _coordinate,
        varname,
        _description,
        block_pos,
        block_len,
        _timezone,
        year,
        month,
        day,
        hour,
        minute,
        second,
        _obs_seconds,
        _obs_dates,
        _gen_dates,
        _gen_seconds,
        edge_s,
        edge_w,
        edge_n,
        edge_e,
        _cx,
        _cy,
        nx,
        ny,
        dx,
        dy,
        _height,
        compress,
        _num_radars,
        _unzip_bytes,
        scale,
        _unused,
        _rgn_id,
        _units,
        _reserved,
    ) = values

    if not label.startswith(b"MOC"):
        raise ValueError(f"{path} is not a MOC mosaic file")

    block = content[block_pos : block_pos + block_len]
    if compress == 1:
        block = bz2.decompress(block)
    elif compress != 0:
        raise ValueError(f"Unsupported compression flag {compress} in {path}")

    raw = np.frombuffer(block, dtype="<i2")
    expected = nx * ny
    if raw.size != expected:
        raise ValueError(f"Unexpected grid size in {path}: {raw.size} != {expected}")

    data = raw.reshape(ny, nx).astype(np.float32)
    data[data <= -29000] = np.nan
    data = data / float(scale if scale else 1)
    data[data < -100] = np.nan
    data = np.nan_to_num(data, nan=fill_value)

    return MosaicGrid(
        data=data,
        product=_decode_ascii(varname),
        obs_time=datetime(year, month, day, hour, minute, second),
        edge_w=edge_w / 1000.0,
        edge_s=edge_s / 1000.0,
        edge_e=edge_e / 1000.0,
        edge_n=edge_n / 1000.0,
        dx=dx / 10000.0,
        dy=dy / 10000.0,
        scale=float(scale),
        path=path,
    )


def find_product_files(input_dir: str, product: str = "QREF") -> List[str]:
    pattern = os.path.join(input_dir, "ZW_MOC", "*", "*", product, "*.bin")
    return sorted(glob.glob(pattern))


def group_product_files_by_time(input_dir: str, product: str) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for path in find_product_files(input_dir, product):
        grouped.setdefault(parse_time_from_name(path), []).append(path)
    return {time: sorted(paths) for time, paths in sorted(grouped.items())}


def pixel_to_lonlat(grid: MosaicGrid, x: float, y: float) -> Tuple[float, float]:
    lon = grid.edge_w + x * grid.dx
    lat = grid.edge_n - y * grid.dy
    return round(float(lon), 5), round(float(lat), 5)


def parse_time_from_name(path: str) -> str:
    name = os.path.basename(path)
    match = re.search(r"_(\d{8})_(\d{6})(?:_\d+)?\.bin$", name)
    if match:
        return match.group(1) + match.group(2)
    grid = read_mosaic_bin(path)
    return grid.obs_time.strftime("%Y%m%d%H%M%S")


def resize_for_model(frame: np.ndarray, size: int, max_dbz: float = 80.0) -> np.ndarray:
    clipped = np.clip(frame, 0.0, max_dbz) / max_dbz
    return cv2.resize(clipped, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)


def iter_sequences(paths: Iterable[str], seq_len: int) -> Iterable[List[str]]:
    paths = list(paths)
    for start in range(0, len(paths) - seq_len + 1):
        yield paths[start : start + seq_len]
