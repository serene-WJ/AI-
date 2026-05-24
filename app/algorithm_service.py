from __future__ import annotations

import math
from statistics import median
from typing import Any

from app.schemas import (
    CleanedKeypoint,
    HeartRateSafety,
    JointAngles,
    PoseAlgorithmRequest,
    PoseAlgorithmResult,
    PoseCandidate,
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
        self.states: dict[str, dict[str, Any]] = {}

    def analyze(self, request: PoseAlgorithmRequest) -> PoseAlgorithmResult:
        state = self._state(request.exercise)
        raw_keypoints = self._select_person_keypoints(request, state)
        cleaning = self._clean_pose(raw_keypoints, state)
        angles = self._calculate_angles(cleaning)
        stage, primary_value = self._recognize_stage(request.exercise, angles, cleaning, state)
        completed_rep = self._update_counter(request.exercise, stage, state)
        quality = self._score_quality(request.exercise, cleaning, angles, stage, state)
        heart_rate_safety = (
            self._check_heart_rate(request.age, request.heart_rate)
            if request.heart_rate is not None
            else None
        )
        training_load = self._assess_training_load(
            quality=quality,
            heart_rate_safety=heart_rate_safety,
            current_reps=state["rep_count"],
            request=request,
        )

        state["previous_stage"] = stage if stage != "unknown" else state["previous_stage"]
        if primary_value is not None:
            previous_primary = state.get("previous_primary")
            state["last_primary_delta"] = (
                abs(primary_value - previous_primary) if previous_primary is not None else None
            )
            state["previous_primary"] = primary_value

        return PoseAlgorithmResult(
            exercise=request.exercise,
            stage=stage,
            rep_count=state["rep_count"],
            completed_rep=completed_rep,
            cleaning=cleaning,
            angles=angles,
            quality=quality,
            heart_rate_safety=heart_rate_safety,
            training_load=training_load,
            algorithm_context=self._build_algorithm_context(
                exercise=request.exercise,
                stage=stage,
                completed_rep=completed_rep,
                cleaning=cleaning,
                angles=angles,
                quality=quality,
                heart_rate_safety=heart_rate_safety,
                training_load=training_load,
                reps=state["rep_count"],
            ),
        )

    def _state(self, exercise: str) -> dict[str, Any]:
        if exercise not in self.states:
            self.states[exercise] = {
                "previous_keypoints": {},
                "previous_stage": "unknown",
                "stage_path": [],
                "rep_count": 0,
                "previous_primary": None,
                "last_primary_delta": None,
                "tracked_center": None,
                "last_high_knee_side": None,
            }
        return self.states[exercise]

    def _select_person_keypoints(
        self,
        request: PoseAlgorithmRequest,
        state: dict[str, Any],
    ) -> dict[str, RawKeypoint]:
        if not request.pose_candidates:
            return request.keypoints

        previous_center = state.get("tracked_center")
        selected = max(
            request.pose_candidates,
            key=lambda candidate: self._candidate_score(candidate, previous_center),
        )
        center = _pose_center(selected.keypoints)
        if center is not None:
            state["tracked_center"] = center
        return selected.keypoints

    def _candidate_score(
        self,
        candidate: PoseCandidate,
        previous_center: tuple[float, float] | None,
    ) -> float:
        valid = [point for point in candidate.keypoints.values() if point.confidence >= LOW_CONFIDENCE_THRESHOLD]
        confidence = sum(point.confidence for point in valid) / len(valid) if valid else 0.0
        score = confidence + len(valid) * 0.05 + candidate.confidence

        if candidate.bbox:
            x1, y1, x2, y2 = candidate.bbox
            score += abs(x2 - x1) * abs(y2 - y1) * 0.00001

        center = _pose_center(candidate.keypoints)
        if center and previous_center:
            score -= _point_distance(center, previous_center) * 0.8
        return score

    def _clean_pose(
        self,
        raw_keypoints: dict[str, RawKeypoint],
        state: dict[str, Any],
    ) -> PoseCleaningResult:
        previous_keypoints: dict[str, CleanedKeypoint] = state["previous_keypoints"]
        cleaned: dict[str, CleanedKeypoint] = {}
        dropped: list[str] = []
        interpolated: list[str] = []

        for name, point in raw_keypoints.items():
            previous = previous_keypoints.get(name)
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

        for name, previous in previous_keypoints.items():
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

        abnormal_frame = self._is_abnormal_frame(cleaned, previous_keypoints)
        if abnormal_frame and previous_keypoints:
            cleaned = previous_keypoints
        else:
            state["previous_keypoints"] = cleaned

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

    def _is_abnormal_frame(
        self,
        cleaned: dict[str, CleanedKeypoint],
        previous_keypoints: dict[str, CleanedKeypoint],
    ) -> bool:
        if not previous_keypoints:
            return False

        movements: list[float] = []
        max_coordinate = 0.0
        for name, point in cleaned.items():
            max_coordinate = max(max_coordinate, abs(point.x), abs(point.y))
            previous = previous_keypoints.get(name)
            if previous and previous.valid and point.valid:
                movements.append(_distance(previous, point))

        if len(movements) < 3:
            return False

        body_scale = self._body_scale(cleaned)
        threshold = 0.35 if max_coordinate <= 2 else max(120.0, body_scale * 0.6)
        return median(movements) > threshold

    def _select_body_side(self, keypoints: dict[str, CleanedKeypoint]) -> str:
        scores: dict[str, float] = {}
        for side in ("left", "right"):
            side_points = [keypoints.get(f"{side}_{joint}") for joint in SIDE_JOINTS]
            valid_points = [point for point in side_points if _is_valid(point)]
            scores[side] = len(valid_points) * 2 + sum(point.confidence for point in valid_points)

        if scores["left"] == 0 and scores["right"] == 0:
            return "unknown"
        return "left" if scores["left"] >= scores["right"] else "right"

    def _body_scale(self, keypoints: dict[str, CleanedKeypoint]) -> float:
        distances = []
        for side in ("left", "right"):
            shoulder = keypoints.get(f"{side}_shoulder")
            hip = keypoints.get(f"{side}_hip")
            knee = keypoints.get(f"{side}_knee")
            ankle = keypoints.get(f"{side}_ankle")
            distances.extend(
                _distance(a, b)
                for a, b in ((shoulder, hip), (hip, knee), (knee, ankle))
                if _is_valid(a) and _is_valid(b)
            )
        return sum(distances)

    def _calculate_angles(self, cleaning: PoseCleaningResult) -> JointAngles:
        keypoints = cleaning.keypoints
        left_knee = _angle(keypoints.get("left_hip"), keypoints.get("left_knee"), keypoints.get("left_ankle"))
        right_knee = _angle(keypoints.get("right_hip"), keypoints.get("right_knee"), keypoints.get("right_ankle"))
        left_hip = _angle(keypoints.get("left_shoulder"), keypoints.get("left_hip"), keypoints.get("left_knee"))
        right_hip = _angle(keypoints.get("right_shoulder"), keypoints.get("right_hip"), keypoints.get("right_knee"))
        left_elbow = _angle(keypoints.get("left_shoulder"), keypoints.get("left_elbow"), keypoints.get("left_wrist"))
        right_elbow = _angle(keypoints.get("right_shoulder"), keypoints.get("right_elbow"), keypoints.get("right_wrist"))
        left_shoulder = _angle(keypoints.get("left_elbow"), keypoints.get("left_shoulder"), keypoints.get("left_hip"))
        right_shoulder = _angle(keypoints.get("right_elbow"), keypoints.get("right_shoulder"), keypoints.get("right_hip"))

        side = cleaning.selected_side
        shoulder = keypoints.get(f"{side}_shoulder") if side != "unknown" else _midpoint(keypoints, "shoulder")
        hip = keypoints.get(f"{side}_hip") if side != "unknown" else _midpoint(keypoints, "hip")
        ankle = keypoints.get(f"{side}_ankle") if side != "unknown" else _midpoint(keypoints, "ankle")

        return JointAngles(
            knee_angle=_round_optional(_avg_defined(left_knee, right_knee)),
            hip_angle=_round_optional(_avg_defined(left_hip, right_hip)),
            trunk_angle=_round_optional(_angle(shoulder, hip, ankle)),
            trunk_forward_lean=_round_optional(_trunk_forward_lean(shoulder, hip)),
            left_knee_angle=_round_optional(left_knee),
            right_knee_angle=_round_optional(right_knee),
            left_hip_angle=_round_optional(left_hip),
            right_hip_angle=_round_optional(right_hip),
            left_elbow_angle=_round_optional(left_elbow),
            right_elbow_angle=_round_optional(right_elbow),
            left_shoulder_angle=_round_optional(left_shoulder),
            right_shoulder_angle=_round_optional(right_shoulder),
            body_line_angle=_round_optional(_body_line_angle(keypoints)),
            stance_width=_round_optional(_stance_width(keypoints)),
        )

    def _recognize_stage(
        self,
        exercise: str,
        angles: JointAngles,
        cleaning: PoseCleaningResult,
        state: dict[str, Any],
    ) -> tuple[str, float | None]:
        if exercise == "squat":
            return self._recognize_bend_stage(angles.knee_angle, state, high=165, low=105)
        if exercise == "push_up":
            return self._recognize_bend_stage(
                _avg_defined(angles.left_elbow_angle, angles.right_elbow_angle),
                state,
                high=155,
                low=95,
                high_stage="up",
                low_stage="bottom",
            )
        if exercise == "lunge":
            front_knee = _min_defined(angles.left_knee_angle, angles.right_knee_angle)
            return self._recognize_bend_stage(front_knee, state, high=160, low=105)
        if exercise == "plank":
            return "holding", angles.body_line_angle
        if exercise == "jumping_jack":
            return self._recognize_jumping_jack(cleaning, angles)
        if exercise == "sit_up":
            return self._recognize_bend_stage(
                angles.hip_angle,
                state,
                high=145,
                low=95,
                high_stage="down",
                low_stage="up",
            )
        if exercise == "high_knees":
            return self._recognize_high_knees(cleaning, state)
        return "unknown", None

    def _recognize_bend_stage(
        self,
        value: float | None,
        state: dict[str, Any],
        high: float,
        low: float,
        high_stage: str = "standing",
        low_stage: str = "bottom",
    ) -> tuple[str, float | None]:
        if value is None:
            return "unknown", None

        previous_value = state.get("previous_primary")
        previous_stage = state.get("previous_stage", "unknown")
        if value >= high:
            return high_stage, value
        if value <= low:
            return low_stage, value
        if previous_value is None:
            return previous_stage, value
        if value < previous_value - 4:
            return "descending", value
        if value > previous_value + 4:
            return "ascending", value
        return previous_stage, value

    def _recognize_jumping_jack(
        self,
        cleaning: PoseCleaningResult,
        angles: JointAngles,
    ) -> tuple[str, float | None]:
        keypoints = cleaning.keypoints
        stance = angles.stance_width
        shoulder_width = _shoulder_width(keypoints)
        hands_up = _hands_above_shoulders(keypoints)
        if stance is None or shoulder_width is None:
            return "unknown", None
        ratio = stance / max(shoulder_width, 0.0001)
        if ratio >= 1.55 and hands_up:
            return "open", ratio
        if ratio <= 1.15 and not hands_up:
            return "closed", ratio
        return "transition", ratio

    def _recognize_high_knees(
        self,
        cleaning: PoseCleaningResult,
        state: dict[str, Any],
    ) -> tuple[str, float | None]:
        keypoints = cleaning.keypoints
        left_hip = keypoints.get("left_hip")
        right_hip = keypoints.get("right_hip")
        left_knee = keypoints.get("left_knee")
        right_knee = keypoints.get("right_knee")
        left_lift = _vertical_lift(left_hip, left_knee)
        right_lift = _vertical_lift(right_hip, right_knee)
        if left_lift is None and right_lift is None:
            return "unknown", None
        if (left_lift or 0) > 0.08 and (left_lift or 0) > (right_lift or 0):
            return "left_knee_up", left_lift
        if (right_lift or 0) > 0.08:
            return "right_knee_up", right_lift
        return "neutral", max(left_lift or 0, right_lift or 0)

    def _update_counter(self, exercise: str, stage: str, state: dict[str, Any]) -> bool:
        if stage == "unknown":
            return False
        if exercise == "plank":
            return False

        previous_stage = state.get("previous_stage", "unknown")
        if stage != previous_stage:
            state["stage_path"].append(stage)
            state["stage_path"] = state["stage_path"][-8:]

        completed = False
        if exercise in {"squat", "lunge"}:
            completed = _contains_ordered_sequence(
                state["stage_path"],
                ["standing", "descending", "bottom", "ascending", "standing"],
            )
        elif exercise == "push_up":
            completed = _contains_ordered_sequence(
                state["stage_path"],
                ["up", "descending", "bottom", "ascending", "up"],
            )
        elif exercise == "sit_up":
            completed = _contains_ordered_sequence(state["stage_path"], ["down", "up", "down"])
        elif exercise == "jumping_jack":
            completed = _contains_ordered_sequence(state["stage_path"], ["closed", "open", "closed"])
        elif exercise == "high_knees":
            last_side = state.get("last_high_knee_side")
            current_side = "left" if stage == "left_knee_up" else "right" if stage == "right_knee_up" else None
            completed = current_side is not None and last_side is not None and current_side != last_side
            if current_side is not None:
                state["last_high_knee_side"] = current_side

        if completed:
            state["rep_count"] += 1
            state["stage_path"] = [stage]
        return completed

    def _score_quality(
        self,
        exercise: str,
        cleaning: PoseCleaningResult,
        angles: JointAngles,
        stage: str,
        state: dict[str, Any],
    ) -> QualityAssessment:
        score = 100
        errors: list[str] = []
        warnings: list[str] = []

        if exercise in {"squat", "lunge"} and self._has_knee_inward(cleaning):
            score -= 20
            errors.append("knee_inward")
        if exercise == "squat" and self._looks_seated(cleaning):
            score -= 15
            warnings.append("seated_or_camera_angle_unclear")
        if exercise in {"squat", "lunge"} and self._shallow_turnaround(stage, state):
            score -= 15
            errors.append("insufficient_depth")
        if exercise in {"squat", "lunge"} and angles.trunk_forward_lean is not None and angles.trunk_forward_lean > 35:
            score -= 15
            errors.append("back_leaning_forward")
        if exercise == "push_up" and self._push_up_hips_sag(cleaning):
            score -= 20
            errors.append("hips_sagging")
        if exercise == "push_up" and self._push_up_shallow(stage, state):
            score -= 15
            errors.append("insufficient_depth")
        if exercise == "plank" and self._plank_body_not_straight(angles):
            score -= 25
            errors.append("body_line_not_straight")
        if exercise == "jumping_jack" and stage == "transition":
            score -= 10
            warnings.append("range_not_clear")
        if exercise == "sit_up" and self._shallow_turnaround(stage, state):
            score -= 15
            errors.append("insufficient_range")
        if exercise == "high_knees" and stage == "neutral":
            score -= 10
            warnings.append("knees_not_high_enough")

        if (state.get("last_primary_delta") or 0) > 28:
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

    def _shallow_turnaround(self, stage: str, state: dict[str, Any]) -> bool:
        return stage == "ascending" and "descending" in state["stage_path"] and "bottom" not in state["stage_path"]

    def _push_up_shallow(self, stage: str, state: dict[str, Any]) -> bool:
        return stage == "ascending" and "descending" in state["stage_path"] and "bottom" not in state["stage_path"]

    def _has_knee_inward(self, cleaning: PoseCleaningResult) -> bool:
        inward = []
        for side, side_sign in (("left", -1), ("right", 1)):
            hip = cleaning.keypoints.get(f"{side}_hip")
            knee = cleaning.keypoints.get(f"{side}_knee")
            ankle = cleaning.keypoints.get(f"{side}_ankle")
            if not (_is_valid(hip) and _is_valid(knee) and _is_valid(ankle)):
                continue
            hip_ankle_x = (hip.x + ankle.x) / 2
            inward.append((knee.x - hip_ankle_x) * side_sign > abs(ankle.x - hip.x) * 0.35)
        return any(inward)

    def _looks_seated(self, cleaning: PoseCleaningResult) -> bool:
        keypoints = cleaning.keypoints
        hip = _midpoint(keypoints, "hip")
        knee = _midpoint(keypoints, "knee")
        ankle = _midpoint(keypoints, "ankle")
        shoulder = _midpoint(keypoints, "shoulder")
        if not (_is_valid(hip) and _is_valid(knee) and _is_valid(ankle) and _is_valid(shoulder)):
            return False
        thigh_nearly_horizontal = abs(hip.y - knee.y) < abs(knee.y - ankle.y) * 0.35
        torso_upright = _trunk_forward_lean(shoulder, hip) is not None and (_trunk_forward_lean(shoulder, hip) or 0) < 15
        return thigh_nearly_horizontal and torso_upright

    def _push_up_hips_sag(self, cleaning: PoseCleaningResult) -> bool:
        shoulder = _midpoint(cleaning.keypoints, "shoulder")
        hip = _midpoint(cleaning.keypoints, "hip")
        ankle = _midpoint(cleaning.keypoints, "ankle")
        if not (_is_valid(shoulder) and _is_valid(hip) and _is_valid(ankle)):
            return False
        line_y = (shoulder.y + ankle.y) / 2
        return hip.y > line_y + max(0.05, abs(ankle.y - shoulder.y) * 0.25)

    def _plank_body_not_straight(self, angles: JointAngles) -> bool:
        return angles.body_line_angle is not None and angles.body_line_angle > 18

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
        exercise: str,
        stage: str,
        completed_rep: bool,
        cleaning: PoseCleaningResult,
        angles: JointAngles,
        quality: QualityAssessment,
        heart_rate_safety: HeartRateSafety | None,
        training_load: TrainingLoadResult,
        reps: int,
    ) -> str:
        lines = [
            "Rule-based sports algorithm result.",
            f"Exercise: {exercise}. Stage: {stage}. Reps: {reps}. Completed rep: {completed_rep}.",
            f"Selected body side: {cleaning.selected_side}. Abnormal frame: {cleaning.abnormal_frame}.",
            f"Angles: knee={angles.knee_angle}, hip={angles.hip_angle}, elbow=({angles.left_elbow_angle},{angles.right_elbow_angle}), body_line={angles.body_line_angle}.",
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


def _point_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


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


def _body_line_angle(keypoints: dict[str, CleanedKeypoint]) -> float | None:
    shoulder = _midpoint(keypoints, "shoulder")
    hip = _midpoint(keypoints, "hip")
    ankle = _midpoint(keypoints, "ankle")
    if not (_is_valid(shoulder) and _is_valid(hip) and _is_valid(ankle)):
        return None
    return 180 - (_angle(shoulder, hip, ankle) or 180)


def _midpoint(keypoints: dict[str, CleanedKeypoint], joint: str) -> CleanedKeypoint | None:
    left = keypoints.get(f"left_{joint}")
    right = keypoints.get(f"right_{joint}")
    valid = [point for point in (left, right) if _is_valid(point)]
    if not valid:
        return None
    return CleanedKeypoint(
        x=sum(point.x for point in valid) / len(valid),
        y=sum(point.y for point in valid) / len(valid),
        confidence=sum(point.confidence for point in valid) / len(valid),
        valid=True,
    )


def _pose_center(keypoints: dict[str, RawKeypoint]) -> tuple[float, float] | None:
    valid = [point for point in keypoints.values() if point.confidence >= LOW_CONFIDENCE_THRESHOLD]
    if not valid:
        return None
    return (
        sum(point.x for point in valid) / len(valid),
        sum(point.y for point in valid) / len(valid),
    )


def _avg_defined(*values: float | None) -> float | None:
    defined = [value for value in values if value is not None]
    if not defined:
        return None
    return sum(defined) / len(defined)


def _min_defined(*values: float | None) -> float | None:
    defined = [value for value in values if value is not None]
    return min(defined) if defined else None


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


def _stance_width(keypoints: dict[str, CleanedKeypoint]) -> float | None:
    left = keypoints.get("left_ankle")
    right = keypoints.get("right_ankle")
    if not (_is_valid(left) and _is_valid(right)):
        return None
    return abs(left.x - right.x)


def _shoulder_width(keypoints: dict[str, CleanedKeypoint]) -> float | None:
    left = keypoints.get("left_shoulder")
    right = keypoints.get("right_shoulder")
    if not (_is_valid(left) and _is_valid(right)):
        return None
    return abs(left.x - right.x)


def _hands_above_shoulders(keypoints: dict[str, CleanedKeypoint]) -> bool:
    left_wrist = keypoints.get("left_wrist")
    right_wrist = keypoints.get("right_wrist")
    left_shoulder = keypoints.get("left_shoulder")
    right_shoulder = keypoints.get("right_shoulder")
    if not all(_is_valid(point) for point in (left_wrist, right_wrist, left_shoulder, right_shoulder)):
        return False
    return left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y


def _vertical_lift(hip: CleanedKeypoint | None, knee: CleanedKeypoint | None) -> float | None:
    if not (_is_valid(hip) and _is_valid(knee)):
        return None
    return hip.y - knee.y


pose_algorithm_engine = PoseAlgorithmEngine()
