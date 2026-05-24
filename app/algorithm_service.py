from __future__ import annotations

import math
from statistics import median

from app.schemas import (
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

LOW_CONFIDENCE_THRESHOLD = 0.3
SMOOTHING_ALPHA = 0.45
SIDE_JOINTS = ("shoulder", "hip", "knee", "ankle")


class PoseAlgorithmEngine:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.previous_keypoints: dict[str, CleanedKeypoint] = {}
        self.previous_knee_angle: float | None = None
        self.last_knee_angle_delta: float | None = None
        self.previous_stage = "unknown"
        self.stage_path: list[str] = []
        self.rep_count = 0

    def analyze(self, request: PoseAlgorithmRequest) -> PoseAlgorithmResult:
        cleaning = self._clean_pose(request.keypoints)
        angles = self._calculate_angles(cleaning)
        stage = self._recognize_squat_stage(angles.knee_angle)
        completed_rep = self._update_counter(stage)
        quality = self._score_quality(cleaning, angles, stage)
        heart_rate_safety = (
            self._check_heart_rate(request.age, request.heart_rate)
            if request.heart_rate is not None
            else None
        )
        training_load = self._assess_training_load(
            quality=quality,
            heart_rate_safety=heart_rate_safety,
            current_reps=self.rep_count,
            request=request,
        )
        algorithm_context = self._build_algorithm_context(
            stage=stage,
            completed_rep=completed_rep,
            cleaning=cleaning,
            angles=angles,
            quality=quality,
            heart_rate_safety=heart_rate_safety,
            training_load=training_load,
        )

        return PoseAlgorithmResult(
            exercise=request.exercise,
            stage=stage,
            rep_count=self.rep_count,
            completed_rep=completed_rep,
            cleaning=cleaning,
            angles=angles,
            quality=quality,
            heart_rate_safety=heart_rate_safety,
            training_load=training_load,
            algorithm_context=algorithm_context,
        )

    def _clean_pose(self, raw_keypoints: dict[str, RawKeypoint]) -> PoseCleaningResult:
        cleaned: dict[str, CleanedKeypoint] = {}
        dropped: list[str] = []
        interpolated: list[str] = []

        for name, point in raw_keypoints.items():
            previous = self.previous_keypoints.get(name)
            if point.confidence < LOW_CONFIDENCE_THRESHOLD:
                dropped.append(name)
                if previous and previous.valid:
                    cleaned[name] = CleanedKeypoint(
                        x=previous.x,
                        y=previous.y,
                        confidence=round(previous.confidence * 0.6, 4),
                        valid=True,
                        interpolated=True,
                    )
                    interpolated.append(name)
                else:
                    cleaned[name] = CleanedKeypoint(
                        x=point.x,
                        y=point.y,
                        confidence=point.confidence,
                        valid=False,
                    )
                continue

            if previous and previous.valid:
                x = previous.x * (1 - SMOOTHING_ALPHA) + point.x * SMOOTHING_ALPHA
                y = previous.y * (1 - SMOOTHING_ALPHA) + point.y * SMOOTHING_ALPHA
            else:
                x = point.x
                y = point.y

            cleaned[name] = CleanedKeypoint(
                x=round(x, 4),
                y=round(y, 4),
                confidence=round(point.confidence, 4),
                valid=True,
            )

        for name, previous in self.previous_keypoints.items():
            if name in cleaned or not previous.valid:
                continue
            cleaned[name] = CleanedKeypoint(
                x=previous.x,
                y=previous.y,
                confidence=round(previous.confidence * 0.5, 4),
                valid=True,
                interpolated=True,
            )
            interpolated.append(name)

        abnormal_frame = self._is_abnormal_frame(cleaned)
        if abnormal_frame and self.previous_keypoints:
            cleaned = self.previous_keypoints

        if not abnormal_frame:
            self.previous_keypoints = cleaned

        valid_points = [point for point in cleaned.values() if point.valid]
        confidence_mean = (
            round(sum(point.confidence for point in valid_points) / len(valid_points), 4)
            if valid_points
            else 0.0
        )

        return PoseCleaningResult(
            keypoints=cleaned,
            selected_side=self._select_body_side(cleaned),
            dropped_keypoints=dropped,
            interpolated_keypoints=interpolated,
            abnormal_frame=abnormal_frame,
            confidence_mean=confidence_mean,
        )

    def _is_abnormal_frame(self, cleaned: dict[str, CleanedKeypoint]) -> bool:
        if not self.previous_keypoints:
            return False

        movements: list[float] = []
        max_coordinate = 0.0
        for name, point in cleaned.items():
            max_coordinate = max(max_coordinate, abs(point.x), abs(point.y))
            previous = self.previous_keypoints.get(name)
            if previous and previous.valid and point.valid:
                movements.append(_distance(previous, point))

        if len(movements) < 3:
            return False

        threshold = 0.35 if max_coordinate <= 2 else max(120.0, self._body_scale(cleaned) * 0.6)
        return median(movements) > threshold

    def _body_scale(self, keypoints: dict[str, CleanedKeypoint]) -> float:
        side = self._select_body_side(keypoints)
        if side == "unknown":
            return 0.0

        shoulder = keypoints.get(f"{side}_shoulder")
        hip = keypoints.get(f"{side}_hip")
        knee = keypoints.get(f"{side}_knee")
        ankle = keypoints.get(f"{side}_ankle")
        pairs = ((shoulder, hip), (hip, knee), (knee, ankle))
        return sum(_distance(a, b) for a, b in pairs if _is_valid(a) and _is_valid(b))

    def _select_body_side(self, keypoints: dict[str, CleanedKeypoint]) -> str:
        scores: dict[str, float] = {}
        for side in ("left", "right"):
            side_points = [keypoints.get(f"{side}_{joint}") for joint in SIDE_JOINTS]
            valid_points = [point for point in side_points if _is_valid(point)]
            scores[side] = len(valid_points) * 2 + sum(point.confidence for point in valid_points)

        if scores["left"] == 0 and scores["right"] == 0:
            return "unknown"
        return "left" if scores["left"] >= scores["right"] else "right"

    def _calculate_angles(self, cleaning: PoseCleaningResult) -> JointAngles:
        side = cleaning.selected_side
        if side == "unknown":
            return JointAngles()

        keypoints = cleaning.keypoints
        shoulder = keypoints.get(f"{side}_shoulder")
        hip = keypoints.get(f"{side}_hip")
        knee = keypoints.get(f"{side}_knee")
        ankle = keypoints.get(f"{side}_ankle")

        knee_angle = _angle(hip, knee, ankle)
        hip_angle = _angle(shoulder, hip, knee)
        trunk_angle = _angle(shoulder, hip, ankle)
        trunk_forward_lean = _trunk_forward_lean(shoulder, hip)

        return JointAngles(
            knee_angle=_round_optional(knee_angle),
            hip_angle=_round_optional(hip_angle),
            trunk_angle=_round_optional(trunk_angle),
            trunk_forward_lean=_round_optional(trunk_forward_lean),
        )

    def _recognize_squat_stage(self, knee_angle: float | None) -> str:
        if knee_angle is None:
            return "unknown"

        previous_angle = self.previous_knee_angle
        self.last_knee_angle_delta = (
            abs(knee_angle - previous_angle) if previous_angle is not None else None
        )
        self.previous_knee_angle = knee_angle

        if knee_angle >= 160:
            return "standing"
        if knee_angle <= 105:
            return "bottom"
        if previous_angle is None:
            return "unknown"
        if knee_angle < previous_angle - 4:
            return "descending"
        if knee_angle > previous_angle + 4:
            return "ascending"
        return self.previous_stage

    def _update_counter(self, stage: str) -> bool:
        completed = False
        if stage == "unknown":
            return False

        if stage != self.previous_stage:
            self.stage_path.append(stage)
            self.stage_path = self.stage_path[-6:]

        if stage == "standing" and self.previous_stage == "ascending":
            completed = _contains_ordered_sequence(
                self.stage_path,
                ["standing", "descending", "bottom", "ascending", "standing"],
            )
            if completed:
                self.rep_count += 1
                self.stage_path = ["standing"]

        self.previous_stage = stage
        return completed

    def _score_quality(
        self,
        cleaning: PoseCleaningResult,
        angles: JointAngles,
        stage: str,
    ) -> QualityAssessment:
        score = 100
        errors: list[str] = []
        warnings: list[str] = []

        if self._has_knee_inward(cleaning):
            score -= 20
            errors.append("knee_inward")

        shallow_turnaround = (
            stage == "ascending"
            and "descending" in self.stage_path
            and "bottom" not in self.stage_path
        )
        if shallow_turnaround:
            score -= 15
            errors.append("insufficient_depth")

        if angles.trunk_forward_lean is not None and angles.trunk_forward_lean > 35:
            score -= 15
            errors.append("back_leaning_forward")

        if self._is_too_fast(angles.knee_angle):
            score -= 10
            warnings.append("movement_too_fast")

        if self._is_left_right_unbalanced(cleaning):
            score -= 10
            warnings.append("left_right_unbalanced")

        if cleaning.confidence_mean < 0.55 or cleaning.abnormal_frame:
            score -= 10
            warnings.append("low_keypoint_confidence")

        return QualityAssessment(
            quality_score=max(0, score),
            errors=errors,
            warnings=warnings,
        )

    def _has_knee_inward(self, cleaning: PoseCleaningResult) -> bool:
        side = cleaning.selected_side
        if side == "unknown":
            return False
        hip = cleaning.keypoints.get(f"{side}_hip")
        knee = cleaning.keypoints.get(f"{side}_knee")
        ankle = cleaning.keypoints.get(f"{side}_ankle")
        if not (_is_valid(hip) and _is_valid(knee) and _is_valid(ankle)):
            return False

        hip_ankle_x = (hip.x + ankle.x) / 2
        side_sign = -1 if side == "left" else 1
        return (knee.x - hip_ankle_x) * side_sign > abs(ankle.x - hip.x) * 0.35

    def _is_too_fast(self, knee_angle: float | None) -> bool:
        if knee_angle is None or self.last_knee_angle_delta is None:
            return False
        return self.last_knee_angle_delta > 28

    def _is_left_right_unbalanced(self, cleaning: PoseCleaningResult) -> bool:
        keypoints = cleaning.keypoints
        left_hip = keypoints.get("left_hip")
        right_hip = keypoints.get("right_hip")
        left_knee = keypoints.get("left_knee")
        right_knee = keypoints.get("right_knee")
        if not all(_is_valid(point) for point in (left_hip, right_hip, left_knee, right_knee)):
            return False
        max_coordinate = max(
            abs(point.x) + abs(point.y)
            for point in (left_hip, right_hip, left_knee, right_knee)
            if point is not None
        )
        threshold = 0.05 if max_coordinate <= 4 else 35
        return abs((left_knee.y - left_hip.y) - (right_knee.y - right_hip.y)) > threshold

    def _check_heart_rate(self, age: int, heart_rate: int) -> HeartRateSafety:
        max_heart_rate = 220 - age
        warning_line = round(max_heart_rate * 0.85)
        stop_line = round(max_heart_rate * 0.9)

        if heart_rate >= stop_line:
            return HeartRateSafety(
                status="stop",
                max_heart_rate=max_heart_rate,
                warning_line=warning_line,
                stop_line=stop_line,
                message="Heart rate is above the stop line. Stop training and recover.",
            )
        if heart_rate >= warning_line:
            return HeartRateSafety(
                status="reduce_intensity",
                max_heart_rate=max_heart_rate,
                warning_line=warning_line,
                stop_line=stop_line,
                message="Heart rate is high. Reduce intensity and watch recovery.",
            )
        return HeartRateSafety(
            status="normal",
            max_heart_rate=max_heart_rate,
            warning_line=warning_line,
            stop_line=stop_line,
            message="Heart rate is within the normal training range.",
        )

    def _assess_training_load(
        self,
        quality: QualityAssessment,
        heart_rate_safety: HeartRateSafety | None,
        current_reps: int,
        request: PoseAlgorithmRequest,
    ) -> TrainingLoadResult:
        context = request.training_context
        risk = 0
        reasons: list[str] = []

        if heart_rate_safety and heart_rate_safety.status == "stop":
            risk += 4
            reasons.append("heart rate reached stop line")
        elif heart_rate_safety and heart_rate_safety.status == "reduce_intensity":
            risk += 2
            reasons.append("heart rate is high")

        if quality.quality_score < 70:
            risk += 2
            reasons.append("movement quality dropped")
        if context.quality_drop_count >= 2:
            risk += 2
            reasons.append("quality declined repeatedly")
        if context.heart_rate_recovery_seconds and context.heart_rate_recovery_seconds > 180:
            risk += 1
            reasons.append("heart rate recovery is slow")
        if context.sleep_quality == "poor":
            risk += 1
            reasons.append("sleep quality is poor")
        if context.duration_minutes >= 60:
            risk += 1
            reasons.append("training duration is long")
        if context.temperature_c is not None and context.temperature_c >= 30:
            risk += 1
            reasons.append("environment is hot")
        if context.humidity is not None and context.humidity >= 75:
            risk += 1
            reasons.append("environment humidity is high")

        if risk >= 4:
            return TrainingLoadResult(
                training_load="high",
                suggestion="rest",
                reason=", ".join(reasons) or "training stress is high",
            )
        if risk >= 2:
            return TrainingLoadResult(
                training_load="medium",
                suggestion="reduce_intensity",
                reason=", ".join(reasons) or "training stress is moderate",
            )
        return TrainingLoadResult(
            training_load="low" if current_reps < 8 else "medium",
            suggestion="continue",
            reason=", ".join(reasons) or "movement quality and safety signals are stable",
        )

    def _build_algorithm_context(
        self,
        stage: str,
        completed_rep: bool,
        cleaning: PoseCleaningResult,
        angles: JointAngles,
        quality: QualityAssessment,
        heart_rate_safety: HeartRateSafety | None,
        training_load: TrainingLoadResult,
    ) -> str:
        lines = [
            "Rule-based sports algorithm result.",
            f"Exercise: squat. Stage: {stage}. Reps: {self.rep_count}. Completed rep: {completed_rep}.",
            f"Selected body side: {cleaning.selected_side}. Abnormal frame: {cleaning.abnormal_frame}.",
            f"Angles: knee={angles.knee_angle}, hip={angles.hip_angle}, trunk={angles.trunk_angle}, lean={angles.trunk_forward_lean}.",
            f"Quality score: {quality.quality_score}. Errors: {quality.errors}. Warnings: {quality.warnings}.",
            f"Training load: {training_load.training_load}. Suggestion: {training_load.suggestion}. Reason: {training_load.reason}.",
        ]
        if heart_rate_safety:
            lines.append(
                f"Heart rate safety: {heart_rate_safety.status}. "
                f"Warning line={heart_rate_safety.warning_line}, stop line={heart_rate_safety.stop_line}."
            )
        return "\n".join(lines)


def _is_valid(point: CleanedKeypoint | None) -> bool:
    return point is not None and point.valid


def _distance(a: CleanedKeypoint | None, b: CleanedKeypoint | None) -> float:
    if a is None or b is None:
        return 0.0
    return math.hypot(a.x - b.x, a.y - b.y)


def _angle(
    a: CleanedKeypoint | None,
    b: CleanedKeypoint | None,
    c: CleanedKeypoint | None,
) -> float | None:
    if not (_is_valid(a) and _is_valid(b) and _is_valid(c)):
        return None

    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)
    ba_len = math.hypot(*ba)
    bc_len = math.hypot(*bc)
    if ba_len == 0 or bc_len == 0:
        return None

    cosine = (ba[0] * bc[0] + ba[1] * bc[1]) / (ba_len * bc_len)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _trunk_forward_lean(
    shoulder: CleanedKeypoint | None,
    hip: CleanedKeypoint | None,
) -> float | None:
    if not (_is_valid(shoulder) and _is_valid(hip)):
        return None
    dx = abs(shoulder.x - hip.x)
    dy = abs(shoulder.y - hip.y)
    if dy == 0:
        return 90.0
    return math.degrees(math.atan(dx / dy))


def _round_optional(value: float | None) -> float | None:
    return round(value, 2) if value is not None else None


def _contains_ordered_sequence(values: list[str], sequence: list[str]) -> bool:
    index = 0
    for value in values:
        if value == sequence[index]:
            index += 1
            if index == len(sequence):
                return True
    return False


pose_algorithm_engine = PoseAlgorithmEngine()
