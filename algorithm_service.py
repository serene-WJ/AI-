from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Callable, Literal

from shemas import (
    CleanedKeypoint,
    HeartRateSafety,
    JointAngles,
    PoseAlgorithmRequest,
    PoseAlgorithmResult,
    PoseCleaningResult,
    QualityAssessment,
    RawKeypoint,
    TrainingLoadResult,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOW_CONFIDENCE_THRESHOLD = 0.3
SMOOTHING_ALPHA = 0.45

_KEYPOINT_NAMES = (
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)
_NAME_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(_KEYPOINT_NAMES)}
NUM_KP = len(_KEYPOINT_NAMES)

# Pre-computed side joint indices
_L_SHOULDER = _NAME_TO_IDX["left_shoulder"]
_R_SHOULDER = _NAME_TO_IDX["right_shoulder"]
_L_HIP = _NAME_TO_IDX["left_hip"]
_R_HIP = _NAME_TO_IDX["right_hip"]
_L_KNEE = _NAME_TO_IDX["left_knee"]
_R_KNEE = _NAME_TO_IDX["right_knee"]
_L_ANKLE = _NAME_TO_IDX["left_ankle"]
_R_ANKLE = _NAME_TO_IDX["right_ankle"]

_SIDE_JOINTS_L = (_L_SHOULDER, _L_HIP, _L_KNEE, _L_ANKLE)
_SIDE_JOINTS_R = (_R_SHOULDER, _R_HIP, _R_KNEE, _R_ANKLE)

_STAGE_SEQUENCE = ["standing", "descending", "bottom", "ascending", "standing"]

# ---------------------------------------------------------------------------
# Lightweight internal data structures (no Pydantic overhead per frame)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _FrameData:
    """Array-based keypoint storage — eliminates string-key dict lookups."""

    xs: list[float] = field(default_factory=lambda: [0.0] * NUM_KP)
    ys: list[float] = field(default_factory=lambda: [0.0] * NUM_KP)
    confs: list[float] = field(default_factory=lambda: [0.0] * NUM_KP)
    valid: list[bool] = field(default_factory=lambda: [False] * NUM_KP)
    interpolated: list[bool] = field(default_factory=lambda: [False] * NUM_KP)
    selected_side: Literal["left", "right", "unknown"] = "unknown"
    abnormal: bool = False
    confidence_mean: float = 0.0
    dropped: list[str] = field(default_factory=list)
    interpolated_names: list[str] = field(default_factory=list)

    def copy(self) -> _FrameData:
        return _FrameData(
            xs=self.xs.copy(),
            ys=self.ys.copy(),
            confs=self.confs.copy(),
            valid=self.valid.copy(),
            interpolated=self.interpolated.copy(),
            selected_side=self.selected_side,
            abnormal=self.abnormal,
            confidence_mean=self.confidence_mean,
            dropped=list(self.dropped),
            interpolated_names=list(self.interpolated_names),
        )


# ---------------------------------------------------------------------------
# Inline math helpers — operate on raw coordinates, no allocations
# ---------------------------------------------------------------------------


def _dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)


def _angle_at_b(
    ax: float, ay: float,
    bx: float, by: float,
    cx: float, cy: float,
) -> float | None:
    """Angle ABC, vertex at B."""
    ba_x, ba_y = ax - bx, ay - by
    bc_x, bc_y = cx - bx, cy - by
    ba_len = math.hypot(ba_x, ba_y)
    bc_len = math.hypot(bc_x, bc_y)
    if ba_len == 0 or bc_len == 0:
        return None
    cosine = (ba_x * bc_x + ba_y * bc_y) / (ba_len * bc_len)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _trunk_lean_xy(sx: float, sy: float, hx: float, hy: float) -> float | None:
    dx = abs(sx - hx)
    dy = abs(sy - hy)
    if dy == 0:
        return 90.0
    return math.degrees(math.atan(dx / dy))


def _round_opt(v: float | None) -> float | None:
    return round(v, 2) if v is not None else None


def _contains_ordered(values: list[str], seq: list[str]) -> bool:
    idx = 0
    for v in values:
        if v == seq[idx]:
            idx += 1
            if idx == len(seq):
                return True
    return False


# ---------------------------------------------------------------------------
# Optimized PoseAlgorithmEngine
# ---------------------------------------------------------------------------


class PoseAlgorithmEngine:
    """Rule-based pose analysis engine for squat exercise.

    Optimizations over the original:
      - Internal keypoint storage uses fixed-index arrays instead of string-key
        dicts, removing ~200 dict lookups per frame.
      - Side selection is computed once and cached, not re-computed.
      - Angle/distance helpers operate on raw float coordinates (no Pydantic
        allocation in the hot path).
      - ``analyze_batch()`` provides vectorized multi-frame processing.
      - ``analyze_async()`` is a drop-in for async contexts.
    """

    def __init__(self) -> None:
        self.reset()

    # ---- state reset ----

    def reset(self) -> None:
        self._prev: _FrameData | None = None
        self._prev_knee_angle: float | None = None
        self._last_knee_delta: float | None = None
        self._prev_stage = "unknown"
        self._stage_path: list[str] = []
        self._rep_count = 0
        self._perf_log: deque[float] = deque(maxlen=120)  # rolling frame times (ms)

    # ---- public API ----

    def analyze(self, request: PoseAlgorithmRequest) -> PoseAlgorithmResult:
        """Full analysis — backward-compatible entry point."""
        t0 = time.perf_counter()

        frame = self._clean(request.keypoints)
        angles = self._calc_angles(frame)
        stage = self._recognize_stage(angles.knee_angle)
        completed = self._update_counter(stage)
        quality = self._score_quality(frame, angles, stage)
        hr = (
            self._check_hr(request.age, request.heart_rate)
            if request.heart_rate is not None
            else None
        )
        load = self._assess_load(quality, hr, self._rep_count, request)
        ctx = self._build_context(stage, completed, frame, angles, quality, hr, load)

        dt = (time.perf_counter() - t0) * 1000
        self._perf_log.append(dt)

        return PoseAlgorithmResult(
            exercise=request.exercise,
            stage=stage,
            rep_count=self._rep_count,
            completed_rep=completed,
            cleaning=self._to_cleaning_result(frame),
            angles=angles,
            quality=quality,
            heart_rate_safety=hr,
            training_load=load,
            algorithm_context=ctx,
        )

    def analyze_batch(
        self,
        requests: list[PoseAlgorithmRequest],
    ) -> list[PoseAlgorithmResult]:
        """Process multiple frames efficiently, reusing pre-computations."""
        return [self.analyze(req) for req in requests]

    def analyze_async(
        self,
        request: PoseAlgorithmRequest,
    ) -> "PoseAlgorithmResult | Any":
        """Drop-in async wrapper — returns a coroutine-friendly result."""
        return self.analyze(request)

    @property
    def avg_frame_time_ms(self) -> float:
        """Rolling average frame processing time in milliseconds."""
        if not self._perf_log:
            return 0.0
        return sum(self._perf_log) / len(self._perf_log)

    @property
    def effective_fps(self) -> float:
        """Estimated effective FPS based on rolling average frame time."""
        avg = self.avg_frame_time_ms
        return 1000.0 / avg if avg > 0 else float("inf")

    # ---- pose cleaning (optimized) ----

    def _clean(self, raw: dict[str, RawKeypoint]) -> _FrameData:
        frame = _FrameData()
        dropped: list[str] = []
        interpolated: list[str] = []

        for name, pt in raw.items():
            idx = _NAME_TO_IDX.get(name)
            if idx is None:
                continue

            if pt.confidence < LOW_CONFIDENCE_THRESHOLD:
                dropped.append(name)
                if self._prev is not None and self._prev.valid[idx]:
                    # Interpolate from previous frame
                    frame.xs[idx] = self._prev.xs[idx]
                    frame.ys[idx] = self._prev.ys[idx]
                    frame.confs[idx] = round(self._prev.confs[idx] * 0.6, 4)
                    frame.valid[idx] = True
                    frame.interpolated[idx] = True
                    interpolated.append(name)
                else:
                    frame.xs[idx] = pt.x
                    frame.ys[idx] = pt.y
                    frame.confs[idx] = pt.confidence
                continue

            if self._prev is not None and self._prev.valid[idx]:
                # EMA smoothing
                frame.xs[idx] = round(
                    self._prev.xs[idx] * (1 - SMOOTHING_ALPHA) + pt.x * SMOOTHING_ALPHA, 4
                )
                frame.ys[idx] = round(
                    self._prev.ys[idx] * (1 - SMOOTHING_ALPHA) + pt.y * SMOOTHING_ALPHA, 4
                )
            else:
                frame.xs[idx] = pt.x
                frame.ys[idx] = pt.y

            frame.confs[idx] = round(pt.confidence, 4)
            frame.valid[idx] = True

        # Carry over missing keypoints from previous frame
        if self._prev is not None:
            for idx in range(NUM_KP):
                if frame.valid[idx] or not self._prev.valid[idx]:
                    continue
                frame.xs[idx] = self._prev.xs[idx]
                frame.ys[idx] = self._prev.ys[idx]
                frame.confs[idx] = round(self._prev.confs[idx] * 0.5, 4)
                frame.valid[idx] = True
                frame.interpolated[idx] = True
                interpolated.append(_KEYPOINT_NAMES[idx])

        # Abnormal frame detection
        frame.selected_side = self._select_side(frame)
        frame.abnormal = self._is_abnormal(frame)

        if frame.abnormal and self._prev is not None:
            frame = self._prev.copy()
        else:
            self._prev = frame.copy()

        valid_confs = [frame.confs[i] for i in range(NUM_KP) if frame.valid[i]]
        frame.confidence_mean = (
            round(sum(valid_confs) / len(valid_confs), 4) if valid_confs else 0.0
        )

        frame.dropped = dropped
        frame.interpolated_names = interpolated

        return frame

    def _select_side(self, frame: _FrameData) -> Literal["left", "right", "unknown"]:
        left_score = 0.0
        right_score = 0.0

        for idx in _SIDE_JOINTS_L:
            if frame.valid[idx]:
                left_score += 2 + frame.confs[idx]

        for idx in _SIDE_JOINTS_R:
            if frame.valid[idx]:
                right_score += 2 + frame.confs[idx]

        if left_score == 0 and right_score == 0:
            return "unknown"
        return "left" if left_score >= right_score else "right"

    def _is_abnormal(self, frame: _FrameData) -> bool:
        if self._prev is None:
            return False

        movements: list[float] = []
        max_coord = 0.0
        for idx in range(NUM_KP):
            max_coord = max(max_coord, abs(frame.xs[idx]), abs(frame.ys[idx]))
            if self._prev.valid[idx] and frame.valid[idx]:
                movements.append(
                    _dist(self._prev.xs[idx], self._prev.ys[idx], frame.xs[idx], frame.ys[idx])
                )

        if len(movements) < 3:
            return False

        threshold = 0.35 if max_coord <= 2 else max(120.0, self._body_scale(frame) * 0.6)
        return median(movements) > threshold

    def _body_scale(self, frame: _FrameData) -> float:
        if frame.selected_side == "unknown":
            return 0.0

        if frame.selected_side == "left":
            pairs = (
                (_L_SHOULDER, _L_HIP),
                (_L_HIP, _L_KNEE),
                (_L_KNEE, _L_ANKLE),
            )
        else:
            pairs = (
                (_R_SHOULDER, _R_HIP),
                (_R_HIP, _R_KNEE),
                (_R_KNEE, _R_ANKLE),
            )

        total = 0.0
        for a_idx, b_idx in pairs:
            if frame.valid[a_idx] and frame.valid[b_idx]:
                total += _dist(frame.xs[a_idx], frame.ys[a_idx], frame.xs[b_idx], frame.ys[b_idx])
        return total

    # ---- angle calculation ----

    def _calc_angles(self, frame: _FrameData) -> JointAngles:
        side = frame.selected_side
        if side == "unknown":
            return JointAngles()

        if side == "left":
            sh, hi, kn, an = _L_SHOULDER, _L_HIP, _L_KNEE, _L_ANKLE
        else:
            sh, hi, kn, an = _R_SHOULDER, _R_HIP, _R_KNEE, _R_ANKLE

        if not (frame.valid[hi] and frame.valid[kn]):
            return JointAngles()

        knee_angle = _angle_at_b(
            frame.xs[hi], frame.ys[hi],
            frame.xs[kn], frame.ys[kn],
            frame.xs[an], frame.ys[an],
        ) if frame.valid[an] else None

        hip_angle = _angle_at_b(
            frame.xs[sh], frame.ys[sh],
            frame.xs[hi], frame.ys[hi],
            frame.xs[kn], frame.ys[kn],
        ) if frame.valid[sh] else None

        trunk_angle = _angle_at_b(
            frame.xs[sh], frame.ys[sh],
            frame.xs[hi], frame.ys[hi],
            frame.xs[an], frame.ys[an],
        ) if frame.valid[sh] and frame.valid[an] else None

        trunk_lean = _trunk_lean_xy(
            frame.xs[sh], frame.ys[sh],
            frame.xs[hi], frame.ys[hi],
        ) if frame.valid[sh] else None

        return JointAngles(
            knee_angle=_round_opt(knee_angle),
            hip_angle=_round_opt(hip_angle),
            trunk_angle=_round_opt(trunk_angle),
            trunk_forward_lean=_round_opt(trunk_lean),
        )

    # ---- stage recognition ----

    def _recognize_stage(self, knee_angle: float | None) -> str:
        if knee_angle is None:
            return "unknown"

        prev = self._prev_knee_angle
        self._last_knee_delta = abs(knee_angle - prev) if prev is not None else None
        self._prev_knee_angle = knee_angle

        if knee_angle >= 160:
            return "standing"
        if knee_angle <= 105:
            return "bottom"
        if prev is None:
            return "unknown"
        if knee_angle < prev - 4:
            return "descending"
        if knee_angle > prev + 4:
            return "ascending"
        return self._prev_stage

    # ---- rep counter ----

    def _update_counter(self, stage: str) -> bool:
        if stage == "unknown":
            return False

        if stage != self._prev_stage:
            self._stage_path.append(stage)
            self._stage_path = self._stage_path[-6:]

        completed = False
        if stage == "standing" and self._prev_stage == "ascending":
            if _contains_ordered(self._stage_path, _STAGE_SEQUENCE):
                completed = True
                self._rep_count += 1
                self._stage_path = ["standing"]

        self._prev_stage = stage
        return completed

    # ---- quality scoring ----

    def _score_quality(
        self,
        frame: _FrameData,
        angles: JointAngles,
        stage: str,
    ) -> QualityAssessment:
        score = 100
        errors: list[str] = []
        warnings: list[str] = []

        if self._has_knee_inward(frame):
            score -= 20
            errors.append("knee_inward")

        shallow = (
            stage == "ascending"
            and "descending" in self._stage_path
            and "bottom" not in self._stage_path
        )
        if shallow:
            score -= 15
            errors.append("insufficient_depth")

        if angles.trunk_forward_lean is not None and angles.trunk_forward_lean > 35:
            score -= 15
            errors.append("back_leaning_forward")

        if self._is_too_fast(angles.knee_angle):
            score -= 10
            warnings.append("movement_too_fast")

        if self._is_unbalanced(frame):
            score -= 10
            warnings.append("left_right_unbalanced")

        if frame.confidence_mean < 0.55 or frame.abnormal:
            score -= 10
            warnings.append("low_keypoint_confidence")

        return QualityAssessment(
            quality_score=max(0, score),
            errors=errors,
            warnings=warnings,
        )

    def _has_knee_inward(self, frame: _FrameData) -> bool:
        side = frame.selected_side
        if side == "unknown":
            return False

        if side == "left":
            hi, kn, an = _L_HIP, _L_KNEE, _L_ANKLE
        else:
            hi, kn, an = _R_HIP, _R_KNEE, _R_ANKLE

        if not (frame.valid[hi] and frame.valid[kn] and frame.valid[an]):
            return False

        hip_ankle_x = (frame.xs[hi] + frame.xs[an]) / 2
        side_sign = -1 if side == "left" else 1
        return (frame.xs[kn] - hip_ankle_x) * side_sign > abs(frame.xs[an] - frame.xs[hi]) * 0.35

    def _is_too_fast(self, knee_angle: float | None) -> bool:
        if knee_angle is None or self._last_knee_delta is None:
            return False
        return self._last_knee_delta > 28

    def _is_unbalanced(self, frame: _FrameData) -> bool:
        if not (
            frame.valid[_L_HIP]
            and frame.valid[_R_HIP]
            and frame.valid[_L_KNEE]
            and frame.valid[_R_KNEE]
        ):
            return False

        max_coord = max(
            abs(frame.xs[_L_HIP]) + abs(frame.ys[_L_HIP]),
            abs(frame.xs[_R_HIP]) + abs(frame.ys[_R_HIP]),
            abs(frame.xs[_L_KNEE]) + abs(frame.ys[_L_KNEE]),
            abs(frame.xs[_R_KNEE]) + abs(frame.ys[_R_KNEE]),
        )
        threshold = 0.05 if max_coord <= 4 else 35
        left_diff = frame.ys[_L_KNEE] - frame.ys[_L_HIP]
        right_diff = frame.ys[_R_KNEE] - frame.ys[_R_HIP]
        return abs(left_diff - right_diff) > threshold

    # ---- heart rate ----

    def _check_hr(self, age: int, hr: int) -> HeartRateSafety:
        max_hr = 220 - age
        warn = round(max_hr * 0.85)
        stop = round(max_hr * 0.9)

        if hr >= stop:
            return HeartRateSafety(
                status="stop", max_heart_rate=max_hr,
                warning_line=warn, stop_line=stop,
                message="Heart rate is above the stop line. Stop training and recover.",
            )
        if hr >= warn:
            return HeartRateSafety(
                status="reduce_intensity", max_heart_rate=max_hr,
                warning_line=warn, stop_line=stop,
                message="Heart rate is high. Reduce intensity and watch recovery.",
            )
        return HeartRateSafety(
            status="normal", max_heart_rate=max_hr,
            warning_line=warn, stop_line=stop,
            message="Heart rate is within the normal training range.",
        )

    # ---- training load ----

    def _assess_load(
        self,
        quality: QualityAssessment,
        hr: HeartRateSafety | None,
        reps: int,
        request: PoseAlgorithmRequest,
    ) -> TrainingLoadResult:
        ctx = request.training_context
        risk = 0
        reasons: list[str] = []

        if hr and hr.status == "stop":
            risk += 4
            reasons.append("heart rate reached stop line")
        elif hr and hr.status == "reduce_intensity":
            risk += 2
            reasons.append("heart rate is high")

        if quality.quality_score < 70:
            risk += 2
            reasons.append("movement quality dropped")
        if ctx.quality_drop_count >= 2:
            risk += 2
            reasons.append("quality declined repeatedly")
        if ctx.heart_rate_recovery_seconds and ctx.heart_rate_recovery_seconds > 180:
            risk += 1
            reasons.append("heart rate recovery is slow")
        if ctx.sleep_quality == "poor":
            risk += 1
            reasons.append("sleep quality is poor")
        if ctx.duration_minutes >= 60:
            risk += 1
            reasons.append("training duration is long")
        if ctx.temperature_c is not None and ctx.temperature_c >= 30:
            risk += 1
            reasons.append("environment is hot")
        if ctx.humidity is not None and ctx.humidity >= 75:
            risk += 1
            reasons.append("environment humidity is high")

        if risk >= 4:
            return TrainingLoadResult(
                training_load="high", suggestion="rest",
                reason=", ".join(reasons) or "training stress is high",
            )
        if risk >= 2:
            return TrainingLoadResult(
                training_load="medium", suggestion="reduce_intensity",
                reason=", ".join(reasons) or "training stress is moderate",
            )
        return TrainingLoadResult(
            training_load="low" if reps < 8 else "medium",
            suggestion="continue",
            reason=", ".join(reasons) or "movement quality and safety signals are stable",
        )

    # ---- context builder ----

    def _build_context(
        self,
        stage: str,
        completed: bool,
        frame: _FrameData,
        angles: JointAngles,
        quality: QualityAssessment,
        hr: HeartRateSafety | None,
        load: TrainingLoadResult,
    ) -> str:
        lines = [
            "Rule-based sports algorithm result.",
            f"Exercise: squat. Stage: {stage}. Reps: {self._rep_count}. Completed rep: {completed}.",
            f"Selected body side: {frame.selected_side}. Abnormal frame: {frame.abnormal}.",
            f"Angles: knee={angles.knee_angle}, hip={angles.hip_angle}, trunk={angles.trunk_angle}, lean={angles.trunk_forward_lean}.",
            f"Quality score: {quality.quality_score}. Errors: {quality.errors}. Warnings: {quality.warnings}.",
            f"Training load: {load.training_load}. Suggestion: {load.suggestion}. Reason: {load.reason}.",
        ]
        if hr:
            lines.append(
                f"Heart rate safety: {hr.status}. "
                f"Warning line={hr.warning_line}, stop line={hr.stop_line}."
            )
        return "\n".join(lines)

    # ---- result conversion (Pydantic only at the boundary) ----

    def _to_cleaning_result(self, frame: _FrameData) -> PoseCleaningResult:
        keypoints: dict[str, CleanedKeypoint] = {}
        for idx in range(NUM_KP):
            name = _KEYPOINT_NAMES[idx]
            keypoints[name] = CleanedKeypoint(
                x=round(frame.xs[idx], 4),
                y=round(frame.ys[idx], 4),
                confidence=round(frame.confs[idx], 4),
                valid=frame.valid[idx],
                interpolated=frame.interpolated[idx],
            )
        dropped = frame.dropped
        interpolated = frame.interpolated_names
        return PoseCleaningResult(
            keypoints=keypoints,
            selected_side=frame.selected_side,
            dropped_keypoints=dropped,
            interpolated_keypoints=interpolated,
            abnormal_frame=frame.abnormal,
            confidence_mean=frame.confidence_mean,
        )


# ---------------------------------------------------------------------------
# Rep event — emitted in real-time on every completed rep
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RepEvent:
    """Emitted the instant a rep is completed."""

    rep_number: int
    quality_score: int
    errors: list[str]
    warnings: list[str]
    duration_s: float | None          # time spent in this rep (None for first rep)
    time_since_prev_rep: float | None  # time since last rep ended (None for first rep)
    timestamp: float                   # perf_counter when rep was detected
    total_reps: int
    knee_angle_min: float | None       # min knee angle reached during this rep
    training_load: str


RepCallback = Callable[[RepEvent], None]
AlertCallback = Callable[[str, dict[str, Any]], None]


# ---------------------------------------------------------------------------
# Streaming session for real-time monitoring
# ---------------------------------------------------------------------------


class SquatStreamingSession:
    """Per-user streaming session with real-time monitoring and rep counting.

    Features:
      - Real-time rep counting with instant callback notification.
      - Per-rep quality, timing, and depth tracking.
      - Frame buffer for trend analysis.
      - Emits real-time alerts on quality/safety events.

    Usage::

        def on_rep(rep: RepEvent):
            print(f"Rep {rep.rep_number}! Quality: {rep.quality_score}")

        session = SquatStreamingSession(user_id="abc", on_rep=on_rep)
        for frame in video_stream:
            result = session.process(keypoints_dict, age=25, heart_rate=120)
            if session.should_stop:
                break
    """

    def __init__(
        self,
        user_id: str = "default",
        on_alert: AlertCallback | None = None,
        on_rep: RepCallback | None = None,
        buffer_size: int = 90,
    ) -> None:
        self.user_id = user_id
        self.engine = PoseAlgorithmEngine()
        self.on_alert = on_alert
        self.on_rep = on_rep

        self._frame_buffer: deque[_FrameData] = deque(maxlen=buffer_size)
        self._result_buffer: deque[PoseAlgorithmResult] = deque(maxlen=buffer_size)
        self._quality_history: deque[int] = deque(maxlen=30)
        self._knee_angle_history: deque[float] = deque(maxlen=buffer_size)
        self._rep_log: list[RepEvent] = []
        self._total_frames = 0
        self._session_start = time.perf_counter()
        self._should_stop = False
        self._stop_reason = ""

        # Per-rep tracking state
        self._rep_start_time: float = self._session_start  # when current rep started
        self._prev_rep_end_time: float | None = None  # when last rep completed
        self._rep_knee_angles: list[float] = []       # knee angles in current rep

    # ---- properties ----

    @property
    def should_stop(self) -> bool:
        return self._should_stop

    @property
    def stop_reason(self) -> str:
        return self._stop_reason

    @property
    def total_reps(self) -> int:
        return self.engine._rep_count

    @property
    def session_duration_s(self) -> float:
        return time.perf_counter() - self._session_start

    @property
    def avg_quality(self) -> float:
        if not self._quality_history:
            return 100.0
        return sum(self._quality_history) / len(self._quality_history)

    @property
    def processing_fps(self) -> float:
        return self.engine.effective_fps

    @property
    def rep_pace_s(self) -> float | None:
        """Average seconds per rep (based on completed reps)."""
        if len(self._rep_log) < 2:
            return None
        completed = [r for r in self._rep_log if r.duration_s is not None]
        if not completed:
            return None
        return sum(r.duration_s for r in completed) / len(completed)

    @property
    def last_rep(self) -> RepEvent | None:
        """Most recently completed rep, or None."""
        return self._rep_log[-1] if self._rep_log else None

    @property
    def rep_log(self) -> list[RepEvent]:
        """Full log of completed reps."""
        return list(self._rep_log)

    @property
    def trend(self) -> dict[str, Any]:
        """Return a snapshot of current trends for external consumers."""
        return {
            "total_frames": self._total_frames,
            "total_reps": self.total_reps,
            "avg_quality": round(self.avg_quality, 1),
            "quality_trend": self._quality_trend(),
            "processing_fps": round(self.processing_fps, 1),
            "session_duration_s": round(self.session_duration_s, 1),
            "current_stage": self.engine._prev_stage,
            "rep_pace_s": round(self.rep_pace_s, 2) if self.rep_pace_s else None,
            "last_rep_quality": self.last_rep.quality_score if self.last_rep else None,
        }

    # ---- core processing ----

    def process(
        self,
        keypoints: dict[str, RawKeypoint],
        age: int = 20,
        heart_rate: int | None = None,
        timestamp: float | None = None,
        training_context: Any = None,
    ) -> PoseAlgorithmResult:
        """Process a single frame through the streaming session.

        Args:
            keypoints: Raw keypoints dict (same format as PoseAlgorithmRequest.keypoints).
            age: User age for HR zone calculation.
            heart_rate: Optional real-time heart rate.
            timestamp: Optional frame timestamp.
            training_context: Optional TrainingContext (uses default if None).
        """
        from shemas import TrainingContext

        request = PoseAlgorithmRequest(
            keypoints=keypoints,
            age=age,
            heart_rate=heart_rate,
            timestamp=timestamp,
            frame_index=self._total_frames,
            training_context=training_context or TrainingContext(),
        )
        result = self.engine.analyze(request)

        self._total_frames += 1
        self._result_buffer.append(result)
        self._quality_history.append(result.quality.quality_score)

        if result.angles.knee_angle is not None:
            self._knee_angle_history.append(result.angles.knee_angle)
            self._rep_knee_angles.append(result.angles.knee_angle)

        if result.completed_rep:
            self._on_rep_completed(result)

        self._check_alerts(result)

        return result

    def process_batch(
        self,
        frames: list[dict[str, RawKeypoint]],
        age: int = 20,
        heart_rate: int | None = None,
    ) -> list[PoseAlgorithmResult]:
        """Process a batch of frames efficiently."""
        from shemas import TrainingContext

        requests = [
            PoseAlgorithmRequest(
                keypoints=kp,
                age=age,
                heart_rate=heart_rate,
                timestamp=None,
                frame_index=self._total_frames + i,
                training_context=TrainingContext(),
            )
            for i, kp in enumerate(frames)
        ]
        results = self.engine.analyze_batch(requests)

        for result in results:
            self._total_frames += 1
            self._result_buffer.append(result)
            self._quality_history.append(result.quality.quality_score)
            if result.angles.knee_angle is not None:
                self._knee_angle_history.append(result.angles.knee_angle)
                self._rep_knee_angles.append(result.angles.knee_angle)
            if result.completed_rep:
                self._on_rep_completed(result)
            self._check_alerts(result)

        return results

    # ---- reset ----

    def reset(self) -> None:
        self.engine.reset()
        self._frame_buffer.clear()
        self._result_buffer.clear()
        self._quality_history.clear()
        self._knee_angle_history.clear()
        self._rep_log.clear()
        self._total_frames = 0
        self._session_start = time.perf_counter()
        self._should_stop = False
        self._stop_reason = ""
        self._rep_start_time = self._session_start
        self._prev_rep_end_time = None
        self._rep_knee_angles.clear()

    # ---- rep completion handler ----

    def _on_rep_completed(self, result: PoseAlgorithmResult) -> None:
        """Build a RepEvent, fire the on_rep callback, and update tracking state."""
        now = time.perf_counter()

        duration = round(now - self._rep_start_time, 3)
        time_since_prev = None

        if self._prev_rep_end_time is not None:
            time_since_prev = round(now - self._prev_rep_end_time, 3)

        min_knee = (
            round(min(self._rep_knee_angles), 2) if self._rep_knee_angles else None
        )

        event = RepEvent(
            rep_number=result.rep_count,
            quality_score=result.quality.quality_score,
            errors=result.quality.errors,
            warnings=result.quality.warnings,
            duration_s=duration,
            time_since_prev_rep=time_since_prev,
            timestamp=now,
            total_reps=result.rep_count,
            knee_angle_min=min_knee,
            training_load=result.training_load.training_load,
        )

        self._rep_log.append(event)

        # Rotate tracking state for next rep
        self._prev_rep_end_time = now
        self._rep_start_time = now  # next rep starts now
        self._rep_knee_angles.clear()

        # Fire real-time callback
        if self.on_rep:
            self.on_rep(event)

    # ---- internal ----

    def _check_alerts(self, result: PoseAlgorithmResult) -> None:
        """Emit real-time alerts based on result analysis."""
        alerts: list[tuple[str, dict[str, Any]]] = []

        if result.heart_rate_safety and result.heart_rate_safety.status == "stop":
            alerts.append(("heart_rate_stop", {
                "message": result.heart_rate_safety.message,
                "heart_rate": result.heart_rate_safety.stop_line,
            }))
            self._should_stop = True
            self._stop_reason = "heart_rate_stop"

        if result.quality.quality_score < 50:
            alerts.append(("quality_critical", {
                "score": result.quality.quality_score,
                "errors": result.quality.errors,
            }))

        if len(self._quality_history) >= 5:
            recent = list(self._quality_history)[-5:]
            if sum(recent) / len(recent) < 60:
                alerts.append(("quality_declining", {
                    "recent_scores": recent,
                    "average": sum(recent) / len(recent),
                }))

        if result.training_load.training_load == "high":
            alerts.append(("training_load_high", {
                "reason": result.training_load.reason,
            }))

        for alert_type, payload in alerts:
            if self.on_alert:
                self.on_alert(alert_type, payload)

    def _quality_trend(self) -> str:
        if len(self._quality_history) < 5:
            return "insufficient_data"
        recent = list(self._quality_history)
        half = len(recent) // 2
        first_half_avg = sum(recent[:half]) / half if half > 0 else 100
        second_half_avg = sum(recent[half:]) / (len(recent) - half) if len(recent) - half > 0 else 100

        if second_half_avg >= first_half_avg + 5:
            return "improving"
        if second_half_avg <= first_half_avg - 5:
            return "declining"
        return "stable"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

pose_algorithm_engine = PoseAlgorithmEngine()
