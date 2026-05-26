import argparse
from dataclasses import dataclass
from typing import Iterable, Optional


MISSING_CSI = 999999.0


def csi(hits: int, false_alarms: int, misses: int, missing_value: float = MISSING_CSI) -> float:
    """Critical Success Index used by the contest.

    CSI = H / (H + F + M). If H + F + M is 0, the contest treats CSI as a
    missing value and excludes it from later aggregation.
    """
    denominator = hits + false_alarms + misses
    if denominator == 0:
        return missing_value
    return hits / denominator


def csi_skill_score(candidate_csi: float, baseline_csi: float) -> float:
    """CSI skill score against the reference algorithm."""
    if candidate_csi == MISSING_CSI:
        return MISSING_CSI
    if baseline_csi >= 1.0:
        return 0.0
    return (candidate_csi - baseline_csi) / (1.0 - baseline_csi)


def lead_time_minutes(event_start_minute: float, first_stable_alarm_minute: float) -> float:
    """Positive means the algorithm recognized the event before it happened."""
    return event_start_minute - first_stable_alarm_minute


def lead_case_score(lead_minutes: float, max_theoretical_lead: float = 60.0) -> float:
    """Single-case lead-time score in [0, 100]."""
    if lead_minutes <= 0:
        return 0.0
    if lead_minutes >= max_theoretical_lead:
        return 100.0
    return lead_minutes / max_theoretical_lead * 100.0


def mean_score(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def lead_skill_score(candidate_mean_score: float, baseline_mean_score: float) -> float:
    """Lead-time skill score against the reference algorithm."""
    if baseline_mean_score >= 100.0:
        return 0.0
    return (candidate_mean_score - baseline_mean_score) / (100.0 - baseline_mean_score)


def average_file_seconds(total_seconds: float, processed_files: int) -> float:
    if processed_files <= 0:
        raise ValueError("processed_files must be positive")
    return total_seconds / processed_files


def efficiency_skill_score(candidate_avg_seconds: float, baseline_avg_seconds: float) -> float:
    """Execution-efficiency skill score against the reference algorithm."""
    if baseline_avg_seconds <= 0:
        raise ValueError("baseline_avg_seconds must be positive")
    return (baseline_avg_seconds - candidate_avg_seconds) / baseline_avg_seconds


def minmax_points(value: float, worst: float, best: float, max_points: float) -> float:
    """Convert a skill value to contest points when all algorithms are known.

    The contest assigns points by comparing the same metric across algorithms in
    the same track. That means exact points require the track-wide best/worst.
    """
    if best == worst:
        return max_points
    return (value - worst) / (best - worst) * max_points


@dataclass
class ContestMetrics:
    hits: int
    false_alarms: int
    misses: int
    candidate_first_alarm_minute: Optional[float] = None
    event_start_minute: Optional[float] = None
    candidate_total_seconds: Optional[float] = None
    processed_files: Optional[int] = None
    baseline_csi: Optional[float] = None
    baseline_lead_score: Optional[float] = None
    baseline_avg_seconds: Optional[float] = None

    def candidate_csi(self) -> float:
        return csi(self.hits, self.false_alarms, self.misses)

    def candidate_csi_skill(self) -> Optional[float]:
        if self.baseline_csi is None:
            return None
        return csi_skill_score(self.candidate_csi(), self.baseline_csi)

    def candidate_lead_minutes(self) -> Optional[float]:
        if self.event_start_minute is None or self.candidate_first_alarm_minute is None:
            return None
        return lead_time_minutes(self.event_start_minute, self.candidate_first_alarm_minute)

    def candidate_lead_score(self) -> Optional[float]:
        lead = self.candidate_lead_minutes()
        if lead is None:
            return None
        return lead_case_score(lead)

    def candidate_lead_skill(self) -> Optional[float]:
        lead_score = self.candidate_lead_score()
        if lead_score is None or self.baseline_lead_score is None:
            return None
        return lead_skill_score(lead_score, self.baseline_lead_score)

    def candidate_avg_seconds(self) -> Optional[float]:
        if self.candidate_total_seconds is None or self.processed_files is None:
            return None
        return average_file_seconds(self.candidate_total_seconds, self.processed_files)

    def candidate_efficiency_skill(self) -> Optional[float]:
        avg_seconds = self.candidate_avg_seconds()
        if avg_seconds is None or self.baseline_avg_seconds is None:
            return None
        return efficiency_skill_score(avg_seconds, self.baseline_avg_seconds)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Contest objective metrics for heavy-rain recognition")
    parser.add_argument("--hits", type=int, required=True)
    parser.add_argument("--false_alarms", type=int, required=True)
    parser.add_argument("--misses", type=int, required=True)
    parser.add_argument("--baseline_csi", type=float)
    parser.add_argument("--event_start_minute", type=float)
    parser.add_argument("--candidate_first_alarm_minute", type=float)
    parser.add_argument("--baseline_lead_score", type=float)
    parser.add_argument("--candidate_total_seconds", type=float)
    parser.add_argument("--processed_files", type=int)
    parser.add_argument("--baseline_avg_seconds", type=float)
    return parser


def main() -> None:
    args = create_parser().parse_args()
    metrics = ContestMetrics(**vars(args))
    print(f"CSI: {metrics.candidate_csi():.6f}")
    if metrics.candidate_csi_skill() is not None:
        print(f"SCSI: {metrics.candidate_csi_skill():.6f}")
    if metrics.candidate_lead_minutes() is not None:
        print(f"Lead minutes: {metrics.candidate_lead_minutes():.2f}")
        print(f"Lead score: {metrics.candidate_lead_score():.6f}")
    if metrics.candidate_lead_skill() is not None:
        print(f"LSS: {metrics.candidate_lead_skill():.6f}")
    if metrics.candidate_avg_seconds() is not None:
        print(f"Avg seconds/file: {metrics.candidate_avg_seconds():.6f}")
    if metrics.candidate_efficiency_skill() is not None:
        print(f"ESS: {metrics.candidate_efficiency_skill():.6f}")


if __name__ == "__main__":
    main()
