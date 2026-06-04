from __future__ import annotations

import argparse
import bz2
import json
import logging
import re
import struct
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage import measure

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import pandas as pd
except ImportError:
    pd = None


NODATA = -32768


@dataclass(frozen=True)
class MosaicGrid:
    path: Path
    timestamp: str
    product: str
    array: np.ndarray
    edge_s: float
    edge_w: float
    edge_n: float
    edge_e: float
    dx: float
    dy: float
    scale: float
    height_m: int


@dataclass
class DerivedFields:
    cref: np.ndarray
    qref: np.ndarray
    top35_m: np.ndarray
    top40_m: np.ndarray
    count35: np.ndarray
    max_column: np.ndarray
    rain_rate: np.ndarray


@dataclass
class CandidateRegion:
    timestamp: str
    lon: float
    lat: float
    polygon: list[list[float]]
    data: float
    features: np.ndarray
    severity: float
    rule_override: bool
    cell_count: int
    mean_rain_rate: float
    max_qref_dbz: float
    component: np.ndarray
    row_start: int
    col_start: int


class HeavyRainBaseline:
    SELECTED_CAP_LEVELS = frozenset({0, 3, 6, 9, 12, 15, 18, 21, 23})
    FEATURE_NAMES = (
        "max_cref_dbz",
        "max_qref_dbz",
        "top35_km",
        "top40_km",
        "mean_count35",
        "max_rain_rate",
        "mean_rain_rate",
        "cell_count",
        "core_pixels",
    )

    def __init__(
        self,
        input_root: Path,
        output_root: Path,
        log_root: Path,
        downsample_factor: int = 4,
        metric_rain_threshold: float = 20.0,
    ) -> None:
        self.input_root = input_root
        self.output_root = output_root
        self.log_root = log_root
        self.downsample_factor = downsample_factor
        self.metric_rain_threshold = metric_rain_threshold
        self.prev_mask: np.ndarray | None = None
        self.prev_cref: np.ndarray | None = None

    def run(self) -> None:
        started = time.perf_counter()
        timestamps = self._collect_timestamps()
        if not timestamps:
            raise RuntimeError("No overlapping CREF/QREF/CAP timestamps were found under the input root.")
        logging.info("Found %d overlapping CREF/QREF/CAP timestamps", len(timestamps))

        frame_candidates: list[tuple[str, list[CandidateRegion]]] = []
        truth_masks: dict[str, np.ndarray] = {}
        for timestamp, files in timestamps:
            print(f"Processing {timestamp} ...")
            logging.info("Processing %s", timestamp)
            derived, template = self._build_fields(timestamp, files)
            truth_masks[timestamp] = derived.rain_rate >= self.metric_rain_threshold
            candidates = self._detect_regions(timestamp, derived, template)
            logging.info("%s candidate_regions=%d", timestamp, len(candidates))
            frame_candidates.append((timestamp, candidates))

        selected_map = self._select_candidates_with_ml(frame_candidates)
        selected_map = self._refine_selected_candidates(frame_candidates, selected_map)
        total_hits = total_false_alarms = total_misses = 0
        json_count = 0
        for timestamp, candidates in frame_candidates:
            selected_candidates = selected_map[timestamp]
            detections = [
                {
                    "lon": round(candidate.lon, 4),
                    "lat": round(candidate.lat, 4),
                    "polygon": candidate.polygon,
                    "data": round(candidate.data, 1),
                }
                for candidate in selected_candidates
            ]
            pred_mask = candidates_to_mask(selected_candidates, truth_masks[timestamp].shape)
            hits, false_alarms, misses = confusion_counts(pred_mask, truth_masks[timestamp])
            total_hits += hits
            total_false_alarms += false_alarms
            total_misses += misses
            out_path = self._write_output(timestamp, detections)
            json_count += 1
            logging.info(
                "%s wrote %s with %d regions hits=%d false_alarms=%d misses=%d CSI=%.6f",
                timestamp,
                out_path,
                len(detections),
                hits,
                false_alarms,
                misses,
                csi(hits, false_alarms, misses),
            )

        total_seconds = time.perf_counter() - started
        total_csi = csi(total_hits, total_false_alarms, total_misses)
        avg_seconds = total_seconds / max(json_count, 1)
        logging.info(
            "METRICS product=QREF+CREF+CAP hits=%d false_alarms=%d misses=%d CSI=%.6f SCSI=N/A LSS=N/A ESS=N/A total_seconds=%.3f processed_files=%d avg_seconds_per_file=%.6f json_count=%d",
            total_hits,
            total_false_alarms,
            total_misses,
            total_csi,
            total_seconds,
            len(timestamps),
            avg_seconds,
            json_count,
        )
        logging.info(
            "SUMMARY {'hits': %d, 'false_alarms': %d, 'misses': %d, 'csi': %.6f, 'json_count': %d}",
            total_hits,
            total_false_alarms,
            total_misses,
            total_csi,
            json_count,
        )

    def _collect_timestamps(self) -> list[tuple[str, dict[str, object]]]:
        zw_root = self.input_root / "ZW_MOC"
        if not zw_root.exists():
            raise RuntimeError(f"Missing ZW_MOC directory: {zw_root}")

        cref_map: dict[str, Path] = {}
        qref_map: dict[str, Path] = {}
        cap_map: dict[str, list[tuple[int, Path]]] = defaultdict(list)

        for path in sorted(zw_root.rglob("*.bin")):
            product = path.parent.name.upper()

            if product == "CREF":
                timestamp = infer_timestamp(path)
                cref_map[timestamp] = path
            elif product == "QREF":
                timestamp = infer_timestamp(path)
                qref_map[timestamp] = path
            elif product == "CAP":
                match = re.search(r"_CAP_(\d{8})_(\d{6})_(\d{2})$", path.stem)
                if not match:
                    continue
                timestamp = f"{match.group(1)}{match.group(2)}"
                level = int(match.group(3))
                cap_map[timestamp].append((level, path))

        common = sorted(set(cref_map) & set(qref_map) & set(cap_map))
        grouped: list[tuple[str, dict[str, object]]] = []
        for timestamp in common:
            grouped.append(
                (
                    timestamp,
                    {
                        "cref": cref_map[timestamp],
                        "qref": qref_map[timestamp],
                        "cap": sorted(cap_map[timestamp], key=lambda item: item[0]),
                    },
                )
            )
        return grouped

    def _build_fields(self, timestamp: str, files: dict[str, object]) -> tuple[DerivedFields, MosaicGrid]:
        cref_grid = read_mosaic(files["cref"], timestamp=timestamp, product="CREF")
        qref_grid = read_mosaic(files["qref"], timestamp=timestamp, product="QREF")

        cref = downsample_max(cref_grid.array, self.downsample_factor, NODATA)
        qref = downsample_max(qref_grid.array, self.downsample_factor, NODATA)

        top35_m = np.zeros_like(cref, dtype=np.int16)
        top40_m = np.zeros_like(cref, dtype=np.int16)
        count35 = np.zeros_like(cref, dtype=np.uint8)
        max_column = np.full_like(cref, NODATA, dtype=np.int16)

        selected_cap_files = [item for item in files["cap"] if item[0] in self.SELECTED_CAP_LEVELS]
        if len(selected_cap_files) < 4:
            selected_cap_files = list(files["cap"])

        for _, cap_path in selected_cap_files:
            cap_grid = read_mosaic(cap_path, timestamp=timestamp, product="CAP")
            layer = downsample_max(cap_grid.array, self.downsample_factor, NODATA)
            valid = layer != NODATA
            max_column[valid] = np.maximum(max_column[valid], layer[valid])

            ge35 = valid & (layer >= 350)
            ge40 = valid & (layer >= 400)
            count35[ge35] += 1
            top35_m[ge35] = np.maximum(top35_m[ge35], cap_grid.height_m)
            top40_m[ge40] = np.maximum(top40_m[ge40], cap_grid.height_m)

        rain_rate = dbz_to_rain_rate(np.maximum(qref, cref) / 10.0)
        rain_rate[(np.maximum(qref, cref) == NODATA)] = 0.0

        derived = DerivedFields(
            cref=cref,
            qref=qref,
            top35_m=top35_m,
            top40_m=top40_m,
            count35=count35,
            max_column=max_column,
            rain_rate=rain_rate,
        )
        return derived, cref_grid

    def _detect_regions(self, timestamp: str, fields: DerivedFields, template: MosaicGrid) -> list[CandidateRegion]:
        cref = fields.cref
        qref = fields.qref
        top35 = fields.top35_m
        top40 = fields.top40_m
        count35 = fields.count35
        rain_rate = fields.rain_rate

        strong_core = (cref >= 420) & (qref >= 350)
        deep_convection = (cref >= 380) & (top35 >= 3500) & (count35 >= 3)
        intense_column = (qref >= 400) & ((top40 >= 3000) | (count35 >= 4))
        growing = np.zeros_like(cref, dtype=bool)
        sustained = np.zeros_like(cref, dtype=bool)

        if self.prev_cref is not None:
            growth = cref - self.prev_cref
            growing = (growth >= 30) & (qref >= 320) & (top35 >= 3000)

        if self.prev_mask is not None:
            influence = ndimage.binary_dilation(self.prev_mask, structure=np.ones((3, 3), dtype=bool))
            sustained = influence & (cref >= 340) & (qref >= 300) & (top35 >= 2500)

        seed_mask = strong_core | deep_convection | intense_column | growing | sustained
        seed_mask &= cref != NODATA

        support_mask = (
            ((qref >= 260) | (cref >= 300) | (rain_rate >= 12.0))
            & ((top35 >= 2000) | (count35 >= 2) | (rain_rate >= 18.0))
        )
        support_mask &= cref != NODATA

        perimeter_mask = (
            ((qref >= 240) | (cref >= 285) | (rain_rate >= 10.0))
            & ((top35 >= 1800) | (count35 >= 1) | (rain_rate >= 15.0))
        )
        perimeter_mask &= cref != NODATA

        seed_labels, seed_num = ndimage.label(seed_mask, structure=np.ones((3, 3), dtype=np.uint8))
        grown_masks: list[np.ndarray] = []
        for seed_id in range(1, seed_num + 1):
            component_seed = seed_labels == seed_id
            if component_seed.sum() < 2:
                continue

            expanded_seed = ndimage.binary_dilation(component_seed, structure=np.ones((5, 5), dtype=bool))
            seed_distance = ndimage.distance_transform_edt(~component_seed)
            near_support = support_mask & (seed_distance <= 5.5)
            near_perimeter = perimeter_mask & (seed_distance <= 3.5)
            local_domain = near_support | near_perimeter | expanded_seed

            grown = ndimage.binary_propagation(expanded_seed, mask=local_domain)
            grown = ndimage.binary_closing(grown, structure=np.ones((7, 7), dtype=bool))

            nearby_fill = perimeter_mask & ndimage.binary_dilation(grown, structure=np.ones((3, 3), dtype=bool))
            nearby_fill &= seed_distance <= 4.5
            grown = grown | nearby_fill
            grown = ndimage.binary_fill_holes(grown)
            grown_masks.append(grown)

        if grown_masks:
            mask = np.logical_or.reduce(grown_masks)
        else:
            mask = seed_mask.copy()

        merge_support = perimeter_mask & ndimage.binary_dilation(seed_mask, structure=np.ones((5, 5), dtype=bool))
        mask = ndimage.binary_closing(mask, structure=np.ones((5, 5), dtype=bool))
        mask = merge_close_components(mask, merge_support, max_gap=2)
        mask = prune_thin_appendages(mask, protected=ndimage.binary_dilation(seed_mask, structure=np.ones((3, 3), dtype=bool)))
        mask = ndimage.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
        mask = ndimage.binary_fill_holes(mask)

        labels, num = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        objects = ndimage.find_objects(labels)

        detections: list[CandidateRegion] = []
        for label_id in range(1, num + 1):
            slc = objects[label_id - 1]
            if slc is None:
                continue

            component = labels[slc] == label_id
            cell_count = int(component.sum())
            if cell_count < 12:
                continue

            cref_comp = fields.cref[slc][component]
            qref_comp = fields.qref[slc][component]
            top35_comp = fields.top35_m[slc][component]
            top40_comp = fields.top40_m[slc][component]
            count35_comp = fields.count35[slc][component]
            rain_comp = fields.rain_rate[slc][component]

            max_cref = int(cref_comp.max())
            max_qref = int(qref_comp.max())
            max_top35 = int(top35_comp.max())
            max_top40 = int(top40_comp.max())
            mean_count35 = float(count35_comp.mean())
            max_rain_rate = float(np.nanmax(rain_comp))
            mean_rain_rate = float(np.nanmean(rain_comp))
            core_pixels = int(((qref_comp >= 350) | (cref_comp >= 420)).sum())

            if max_qref < 360 and max_cref < 400:
                continue
            if max_top35 < 3000 and max_cref < 460:
                continue
            if mean_count35 < 2.0 and max_top40 < 2000:
                continue
            if max_rain_rate < 18.0:
                continue
            if core_pixels < 3 and cell_count < 30:
                continue
            if mean_rain_rate < 10.0 and cell_count < 40:
                continue

            polygon = component_to_polygon(
                component=component,
                row_slice=slc[0],
                col_slice=slc[1],
                grid=template,
                factor=self.downsample_factor,
            )
            if len(polygon) < 4:
                continue

            centroid_row, centroid_col = component_centroid(component, slc)
            lat, lon = cell_center_to_lat_lon(
                row=centroid_row,
                col=centroid_col,
                grid=template,
                factor=self.downsample_factor,
            )

            features = np.array(
                [
                    max_cref / 10.0,
                    max_qref / 10.0,
                    max_top35 / 1000.0,
                    max_top40 / 1000.0,
                    mean_count35,
                    max_rain_rate,
                    mean_rain_rate,
                    float(cell_count),
                    float(core_pixels),
                ],
                dtype=np.float32,
            )
            severity = (
                max_qref / 10.0 * 0.30
                + max_rain_rate * 0.35
                + max_top35 / 1000.0 * 2.0
                + mean_rain_rate * 0.20
                + min(cell_count, 120) * 0.05
            )
            rule_override = (
                max_qref >= 450
                or max_rain_rate >= 55.0
                or (core_pixels >= 8 and max_top35 >= 4000)
            )

            detections.append(
                CandidateRegion(
                    timestamp=timestamp,
                    lon=round(lon, 4),
                    lat=round(lat, 4),
                    polygon=polygon,
                    data=round(max_rain_rate, 1),
                    features=features,
                    severity=float(severity),
                    rule_override=rule_override,
                    cell_count=cell_count,
                    mean_rain_rate=mean_rain_rate,
                    max_qref_dbz=max_qref / 10.0,
                    component=component.copy(),
                    row_start=slc[0].start or 0,
                    col_start=slc[1].start or 0,
                )
            )

        self.prev_mask = mask
        self.prev_cref = fields.cref.copy()
        return detections

    def _select_candidates_with_ml(
        self,
        frame_candidates: list[tuple[str, list[CandidateRegion]]],
    ) -> dict[str, list[CandidateRegion]]:
        selected: dict[str, list[CandidateRegion]] = {timestamp: [] for timestamp, _ in frame_candidates}
        all_candidates = [candidate for _, candidates in frame_candidates for candidate in candidates]
        if len(all_candidates) < 6:
            for timestamp, candidates in frame_candidates:
                selected[timestamp] = candidates
            return selected

        feature_matrix = np.stack([candidate.features for candidate in all_candidates], axis=0)
        train_x, train_y, train_w = self._build_pseudo_training_set(all_candidates)

        if train_x is None:
            raise RuntimeError(
                "LightGBM-only mode could not build a usable pseudo-labeled training set from the current cases."
            )

        model = self._build_classifier()
        train_input = self._model_input(train_x)
        infer_input = self._model_input(feature_matrix)
        model.fit(train_input, train_y, sample_weight=train_w)
        probabilities = model.predict_proba(infer_input)[:, 1]
        train_probabilities = model.predict_proba(train_input)[:, 1]

        positive_scores = train_probabilities[train_y == 1]
        negative_scores = train_probabilities[train_y == 0]
        positive_floor = float(np.quantile(positive_scores, 0.20)) if len(positive_scores) else 0.60
        negative_ceiling = float(np.quantile(negative_scores, 0.85)) if len(negative_scores) else 0.45
        probability_threshold = max(0.50, min(0.75, (positive_floor + negative_ceiling) / 2.0))

        for candidate, probability in zip(all_candidates, probabilities):
            keep = probability >= probability_threshold
            if candidate.rule_override:
                keep = True
            if keep:
                selected[candidate.timestamp].append(candidate)

        for timestamp, candidates in frame_candidates:
            if not selected[timestamp]:
                selected[timestamp] = sorted(candidates, key=lambda item: item.severity, reverse=True)[:1]
            else:
                selected[timestamp].sort(key=lambda item: item.severity, reverse=True)

        return selected

    def _refine_selected_candidates(
        self,
        frame_candidates: list[tuple[str, list[CandidateRegion]]],
        selected_map: dict[str, list[CandidateRegion]],
    ) -> dict[str, list[CandidateRegion]]:
        ordered_timestamps = [timestamp for timestamp, _ in frame_candidates]
        candidate_map = {timestamp: candidates for timestamp, candidates in frame_candidates}
        refined: dict[str, list[CandidateRegion]] = {}

        for idx, timestamp in enumerate(ordered_timestamps):
            prev_selected = selected_map.get(ordered_timestamps[idx - 1], []) if idx > 0 else []
            next_selected = selected_map.get(ordered_timestamps[idx + 1], []) if idx + 1 < len(ordered_timestamps) else []
            current_selected = list(selected_map.get(timestamp, []))

            kept: list[CandidateRegion] = []
            for candidate in sorted(current_selected, key=lambda item: item.severity, reverse=True):
                matched_prev = any(self._is_temporal_match(candidate, other) for other in prev_selected)
                matched_next = any(self._is_temporal_match(candidate, other) for other in next_selected)
                persistent = matched_prev or matched_next
                strong = (
                    candidate.rule_override
                    or candidate.severity >= 68.0
                    or candidate.data >= 32.0
                    or candidate.cell_count >= 44
                    or candidate.max_qref_dbz >= 43.0
                )
                if persistent or strong:
                    kept.append(candidate)

            if idx > 0 and idx + 1 < len(ordered_timestamps):
                for candidate in sorted(candidate_map[timestamp], key=lambda item: item.severity, reverse=True):
                    if candidate in kept:
                        continue
                    matched_prev = any(self._is_temporal_match(candidate, other) for other in prev_selected)
                    matched_next = any(self._is_temporal_match(candidate, other) for other in next_selected)
                    if (
                        matched_prev
                        and matched_next
                        and candidate.severity >= 52.0
                        and candidate.mean_rain_rate >= 14.0
                    ):
                        kept.append(candidate)

            kept = self._suppress_near_duplicates(kept)
            if not kept and current_selected:
                kept = sorted(current_selected, key=lambda item: item.severity, reverse=True)[:1]
            refined[timestamp] = sorted(kept, key=lambda item: item.severity, reverse=True)

        return refined

    def _build_classifier(self):
        if lgb is None:
            raise RuntimeError(
                "LightGBM is required but is not available in the current Python environment. "
                "Install/activate the environment that provides lightgbm, then rerun main.py."
            )
        return lgb.LGBMClassifier(
            objective="binary",
            n_estimators=160,
            learning_rate=0.06,
            max_depth=5,
            num_leaves=24,
            min_child_samples=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=0.5,
            random_state=42,
            n_jobs=1,
            verbosity=-1,
        )

    def _model_input(self, matrix: np.ndarray):
        if pd is None:
            return matrix
        return pd.DataFrame(matrix, columns=self.FEATURE_NAMES)

    def _suppress_near_duplicates(self, candidates: list[CandidateRegion]) -> list[CandidateRegion]:
        kept: list[CandidateRegion] = []
        for candidate in sorted(candidates, key=lambda item: item.severity, reverse=True):
            if any(self._is_duplicate_candidate(candidate, other) for other in kept):
                continue
            kept.append(candidate)
        return kept

    def _is_temporal_match(self, left: CandidateRegion, right: CandidateRegion) -> bool:
        left_box = candidate_bbox(left)
        right_box = candidate_bbox(right)
        iou = bbox_iou(left_box, right_box)
        distance = ((left.lon - right.lon) ** 2 + (left.lat - right.lat) ** 2) ** 0.5
        return iou >= 0.08 or distance <= 1.45

    def _is_duplicate_candidate(self, left: CandidateRegion, right: CandidateRegion) -> bool:
        left_box = candidate_bbox(left)
        right_box = candidate_bbox(right)
        iou = bbox_iou(left_box, right_box)
        distance = ((left.lon - right.lon) ** 2 + (left.lat - right.lat) ** 2) ** 0.5
        return iou >= 0.35 or (iou >= 0.18 and distance <= 0.75)

    def _build_pseudo_training_set(
        self,
        candidates: list[CandidateRegion],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
        if len(candidates) < 8:
            return None, None, None

        severities = np.array([candidate.severity for candidate in candidates], dtype=np.float32)
        severe_cut = float(np.quantile(severities, 0.72))
        weak_cut = float(np.quantile(severities, 0.32))

        train_x: list[np.ndarray] = []
        train_y: list[int] = []
        train_w: list[float] = []

        for candidate in candidates:
            max_cref, max_qref, top35_km, top40_km, mean_count35, max_rain, mean_rain, cell_count, core_pixels = (
                candidate.features.tolist()
            )

            positive = candidate.rule_override or (
                candidate.severity >= severe_cut
                and (max_qref >= 39.0 or max_rain >= 24.0 or top35_km >= 3.4 or core_pixels >= 6.0)
            )
            negative = (
                candidate.severity <= weak_cut
                and not candidate.rule_override
                and max_qref < 48.0
                and max_rain < 55.0
                and top35_km < 5.5
                and core_pixels < 18.0
            )

            if positive and not negative:
                train_x.append(candidate.features)
                train_y.append(1)
                train_w.append(2.5 if candidate.rule_override else 1.5)
            elif negative and not positive:
                train_x.append(candidate.features)
                train_y.append(0)
                train_w.append(1.0)

        positives = sum(label == 1 for label in train_y)
        negatives = sum(label == 0 for label in train_y)
        if positives < 3 or negatives < 3:
            return None, None, None

        return (
            np.stack(train_x, axis=0),
            np.array(train_y, dtype=np.int32),
            np.array(train_w, dtype=np.float32),
        )

    def _write_output(self, timestamp: str, detections: list[dict[str, object]]) -> Path:
        year = timestamp[:4]
        day = timestamp[:8]
        out_dir = self.output_root / "result" / year / day / "heavyrain"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{timestamp}_heavyrain.json"

        payload = {
            "site_code": "",
            "date_time": timestamp,
            "datas": detections,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")
        return out_path


def read_mosaic(path: str | Path, timestamp: str | None = None, product: str | None = None) -> MosaicGrid:
    path = Path(path)
    raw = path.read_bytes()

    edge_s = struct.unpack_from("<i", raw, 124)[0] / 1000.0
    edge_w = struct.unpack_from("<i", raw, 128)[0] / 1000.0
    edge_n = struct.unpack_from("<i", raw, 132)[0] / 1000.0
    edge_e = struct.unpack_from("<i", raw, 136)[0] / 1000.0
    nx = struct.unpack_from("<i", raw, 148)[0]
    ny = struct.unpack_from("<i", raw, 152)[0]
    dx = struct.unpack_from("<i", raw, 156)[0] / 10000.0
    dy = struct.unpack_from("<i", raw, 160)[0] / 10000.0
    height_m = struct.unpack_from("<h", raw, 164)[0]
    compress = struct.unpack_from("<h", raw, 166)[0]
    scale = float(struct.unpack_from("<h", raw, 176)[0])
    block_pos = struct.unpack_from("<i", raw, 88)[0]
    block_len = struct.unpack_from("<i", raw, 92)[0]

    payload = raw[block_pos : block_pos + block_len]
    if compress == 1:
        payload = bz2.decompress(payload)
    elif compress != 0:
        raise RuntimeError(f"Unsupported compression flag {compress} in {path}")

    array = np.frombuffer(payload, dtype="<i2").reshape(ny, nx)
    return MosaicGrid(
        path=path,
        timestamp=timestamp or infer_timestamp(path),
        product=product or infer_product(path),
        array=array,
        edge_s=edge_s,
        edge_w=edge_w,
        edge_n=edge_n,
        edge_e=edge_e,
        dx=dx,
        dy=dy,
        scale=scale,
        height_m=height_m,
    )


def infer_product(path: Path) -> str:
    return path.parent.name.upper()


def infer_timestamp(path: Path) -> str:
    match = re.search(r"_(\d{8})_(\d{6})(?:_(\d{2}))?$", path.stem)
    if not match:
        raise RuntimeError(f"Cannot infer timestamp from {path.name}")
    return f"{match.group(1)}{match.group(2)}"


def downsample_max(array: np.ndarray, factor: int, nodata: int) -> np.ndarray:
    ny, nx = array.shape
    pad_y = (-ny) % factor
    pad_x = (-nx) % factor
    if pad_y or pad_x:
        array = np.pad(array, ((0, pad_y), (0, pad_x)), constant_values=nodata)
    coarse = array.reshape(array.shape[0] // factor, factor, array.shape[1] // factor, factor).max(axis=(1, 3))
    return coarse


def dbz_to_rain_rate(dbz: np.ndarray) -> np.ndarray:
    safe = np.clip(dbz, a_min=0.0, a_max=75.0)
    z = np.power(10.0, safe / 10.0)
    rain = np.power(z / 300.0, 1.0 / 1.4)
    return np.clip(rain, a_min=0.0, a_max=200.0)


def component_centroid(component: np.ndarray, slc: tuple[slice, slice]) -> tuple[int, int]:
    rows, cols = np.nonzero(component)
    row = int(round(rows.mean())) + (slc[0].start or 0)
    col = int(round(cols.mean())) + (slc[1].start or 0)
    return row, col


def cell_center_to_lat_lon(row: int, col: int, grid: MosaicGrid, factor: int) -> tuple[float, float]:
    coarse_dx = grid.dx * factor
    coarse_dy = grid.dy * factor
    lat = grid.edge_n - (row + 0.5) * coarse_dy
    lon = grid.edge_w + (col + 0.5) * coarse_dx
    return lat, lon


def candidate_bbox(candidate: CandidateRegion) -> tuple[float, float, float, float]:
    xs = [float(point[0]) for point in candidate.polygon]
    ys = [float(point[1]) for point in candidate.polygon]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    left_w, left_s, left_e, left_n = left
    right_w, right_s, right_e, right_n = right

    inter_w = max(left_w, right_w)
    inter_s = max(left_s, right_s)
    inter_e = min(left_e, right_e)
    inter_n = min(left_n, right_n)
    if inter_e <= inter_w or inter_n <= inter_s:
        return 0.0

    inter_area = (inter_e - inter_w) * (inter_n - inter_s)
    left_area = max(0.0, left_e - left_w) * max(0.0, left_n - left_s)
    right_area = max(0.0, right_e - right_w) * max(0.0, right_n - right_s)
    union_area = left_area + right_area - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def candidates_to_mask(candidates: list[CandidateRegion], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for candidate in candidates:
        rows, cols = candidate.component.shape
        row_end = min(candidate.row_start + rows, shape[0])
        col_end = min(candidate.col_start + cols, shape[1])
        if row_end <= candidate.row_start or col_end <= candidate.col_start:
            continue
        local_rows = row_end - candidate.row_start
        local_cols = col_end - candidate.col_start
        mask[candidate.row_start:row_end, candidate.col_start:col_end] |= candidate.component[:local_rows, :local_cols]
    return mask


def confusion_counts(pred_mask: np.ndarray, target_mask: np.ndarray) -> tuple[int, int, int]:
    hits = int(np.logical_and(pred_mask, target_mask).sum())
    false_alarms = int(np.logical_and(pred_mask, ~target_mask).sum())
    misses = int(np.logical_and(~pred_mask, target_mask).sum())
    return hits, false_alarms, misses


def csi(hits: int, false_alarms: int, misses: int) -> float:
    denominator = hits + false_alarms + misses
    if denominator == 0:
        return 999999.0
    return hits / denominator


def prune_thin_appendages(mask: np.ndarray, protected: np.ndarray, iterations: int = 2) -> np.ndarray:
    pruned = mask.copy()
    kernel = np.ones((3, 3), dtype=np.uint8)
    protected = protected.astype(bool)

    for _ in range(iterations):
        neighbor_count = ndimage.convolve(pruned.astype(np.uint8), kernel, mode="constant", cval=0)
        removable = pruned & ~protected & (neighbor_count <= 3)
        if not np.any(removable):
            break
        pruned[removable] = False

    return pruned


def component_to_polygon(
    component: np.ndarray,
    row_slice: slice,
    col_slice: slice,
    grid: MosaicGrid,
    factor: int,
) -> list[list[float]]:
    if not np.any(component):
        return []

    padded = np.pad(component.astype(np.uint8), 1, mode="constant")
    contours = measure.find_contours(padded, 0.5)
    if not contours:
        return bounding_box_polygon(component, row_slice, col_slice, grid, factor)

    contour = max(contours, key=len)
    sample_step = max(1, len(contour) // 120)
    contour = contour[::sample_step]

    base_row = row_slice.start or 0
    base_col = col_slice.start or 0
    cell_dx = grid.dx * factor
    cell_dy = grid.dy * factor

    polygon: list[list[float]] = []
    for local_row, local_col in contour:
        row_edge = base_row + (float(local_row) - 1.0)
        col_edge = base_col + (float(local_col) - 1.0)
        lat = grid.edge_n - row_edge * cell_dy
        lon = grid.edge_w + col_edge * cell_dx
        polygon.append([round(float(lon), 4), round(float(lat), 4)])

    if len(polygon) < 4:
        return bounding_box_polygon(component, row_slice, col_slice, grid, factor)
    if polygon[0] != polygon[-1]:
        polygon.append(polygon[0])
    return polygon


def bounding_box_polygon(
    component: np.ndarray,
    row_slice: slice,
    col_slice: slice,
    grid: MosaicGrid,
    factor: int,
) -> list[list[float]]:
    rows, cols = np.nonzero(component)
    if len(rows) == 0:
        return []

    min_row = (row_slice.start or 0) + int(rows.min())
    max_row = (row_slice.start or 0) + int(rows.max()) + 1
    min_col = (col_slice.start or 0) + int(cols.min())
    max_col = (col_slice.start or 0) + int(cols.max()) + 1

    cell_dx = grid.dx * factor
    cell_dy = grid.dy * factor
    north = grid.edge_n - min_row * cell_dy
    south = grid.edge_n - max_row * cell_dy
    west = grid.edge_w + min_col * cell_dx
    east = grid.edge_w + max_col * cell_dx

    return [
        [round(west, 4), round(north, 4)],
        [round(east, 4), round(north, 4)],
        [round(east, 4), round(south, 4)],
        [round(west, 4), round(south, 4)],
        [round(west, 4), round(north, 4)],
    ]


def merge_close_components(mask: np.ndarray, bridge_mask: np.ndarray, max_gap: int) -> np.ndarray:
    if max_gap <= 0:
        return mask

    merged = mask.copy()
    structure = np.ones((3, 3), dtype=np.uint8)

    for _ in range(max_gap):
        labels, num = ndimage.label(merged, structure=structure)
        if num <= 1:
            return merged

        dilated = ndimage.grey_dilation(labels, footprint=np.ones((3, 3), dtype=np.int16))
        candidates = (labels == 0) & (dilated > 0) & bridge_mask
        if not np.any(candidates):
            break
        merged = merged | candidates

    merged = ndimage.binary_fill_holes(merged)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU baseline for group heavy-rain detection.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data"),
        help="Input root that contains ZW_MOC, flash, radar, satellite and sounding directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="Output root. Results are written under output/result/YYYY/YYYYMMDD/heavyrain/.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("log"),
        help="Log directory. A timestamped run log is written here.",
    )
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=4,
        help="Max-pooling factor for nationwide mosaics. Larger values are faster but coarser.",
    )
    parser.add_argument(
        "--metric-rain-threshold",
        type=float,
        default=20.0,
        help="Rain-rate threshold in mm/h used for self-check CSI logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.log.mkdir(parents=True, exist_ok=True)
    log_path = args.log / f"heavy-rain-{datetime.utcnow():%Y%m%d%H%M%S}.log"
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info(
        "Algorithm=HeavyRainBaseline input=%s output=%s log=%s downsample_factor=%d metric_rain_threshold=%.3f",
        args.input,
        args.output,
        args.log,
        args.downsample_factor,
        args.metric_rain_threshold,
    )
    baseline = HeavyRainBaseline(
        input_root=args.input,
        output_root=args.output,
        log_root=args.log,
        downsample_factor=args.downsample_factor,
        metric_rain_threshold=args.metric_rain_threshold,
    )
    try:
        baseline.run()
    except Exception:
        logging.exception("Algorithm failed")
        raise
    print(f"log written to {log_path}")


if __name__ == "__main__":
    main()
