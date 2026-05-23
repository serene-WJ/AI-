from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ImageSize(BaseModel):
    width: int
    height: int


class NormalizedPoint(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class NormalizedBox(BaseModel):
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)


class SpatialRelation(BaseModel):
    horizontal: Literal["left", "center", "right"]
    vertical: Literal["top", "middle", "bottom"]
    size: Literal["small", "medium", "large"]
    distance_hint: Literal["far", "medium", "near"]


class DetectionObject(BaseModel):
    object_id: int
    label: str
    confidence: float
    bbox: NormalizedBox
    center: NormalizedPoint
    area_ratio: float
    spatial: SpatialRelation


class SceneObservation(BaseModel):
    timestamp: float
    source: str
    image_size: ImageSize
    object_count: int
    counts: dict[str, int]
    sports_hints: dict[str, Any]
    objects: list[DetectionObject]
    summary: str
    llm_context: str


class YoloDetectionInput(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[float] = Field(min_length=4, max_length=4)
    normalized: bool = False
    object_id: int | None = None


class YoloSceneIngestRequest(BaseModel):
    image_size: ImageSize
    detections: list[YoloDetectionInput]
    source: str = "external_yolo"
    timestamp: float | None = None


class YoloSceneIngestResponse(BaseModel):
    scene: SceneObservation


class AskRequest(BaseModel):
    question: str
    include_scene_json: bool = True


class AskResponse(BaseModel):
    answer: str
    scene: SceneObservation | None


class DetectResponse(BaseModel):
    scene: SceneObservation


class HealthResponse(BaseModel):
    ok: bool
    model_loaded: bool
    model_path: str
    details: dict[str, Any] = Field(default_factory=dict)


class RawKeypoint(BaseModel):
    x: float
    y: float
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def parse_list_keypoint(cls, data: Any) -> Any:
        if isinstance(data, (list, tuple)):
            if len(data) == 2:
                return {"x": data[0], "y": data[1], "confidence": 1.0}
            if len(data) == 3:
                return {"x": data[0], "y": data[1], "confidence": data[2]}
        return data


class CleanedKeypoint(RawKeypoint):
    valid: bool = True
    interpolated: bool = False


class JointAngles(BaseModel):
    knee_angle: float | None = None
    hip_angle: float | None = None
    trunk_angle: float | None = None
    trunk_forward_lean: float | None = None


class PoseCleaningResult(BaseModel):
    keypoints: dict[str, CleanedKeypoint]
    selected_side: Literal["left", "right", "unknown"]
    dropped_keypoints: list[str] = Field(default_factory=list)
    interpolated_keypoints: list[str] = Field(default_factory=list)
    abnormal_frame: bool = False
    confidence_mean: float = 0.0


class QualityAssessment(BaseModel):
    quality_score: int
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class HeartRateSafety(BaseModel):
    status: Literal["normal", "reduce_intensity", "stop"]
    max_heart_rate: int
    warning_line: int
    stop_line: int
    message: str


class TrainingContext(BaseModel):
    current_sets: int = 0
    current_reps: int = 0
    quality_drop_count: int = 0
    heart_rate_recovery_seconds: int | None = None
    sleep_quality: Literal["poor", "fair", "good"] = "fair"
    duration_minutes: float = 0.0
    temperature_c: float | None = None
    humidity: float | None = None


class WatchHealthData(BaseModel):
    user_id: str = "default"
    session_id: str | None = None
    age: int = Field(default=20, ge=1, le=120)
    heart_rate: int | None = Field(default=None, ge=30, le=240)
    sleep_quality: Literal["poor", "fair", "good"] = "fair"
    sleep_hours: float | None = Field(default=None, ge=0.0, le=24.0)
    heart_rate_recovery_seconds: int | None = None
    timestamp: float | None = None


class WatchHealthResponse(BaseModel):
    health: WatchHealthData | None


class HealthKitSample(BaseModel):
    user_id: str = "default"
    session_id: str | None = None
    device_id: str | None = None
    sample_type: Literal[
        "heart_rate",
        "resting_heart_rate",
        "heart_rate_recovery",
        "sleep",
        "sleep_quality",
    ]
    value: float | str
    unit: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    source: str = "apple_watch"
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthKitImportRequest(BaseModel):
    user_id: str = "default"
    session_id: str | None = None
    device_id: str | None = None
    age: int = Field(default=20, ge=1, le=120)
    samples: list[HealthKitSample]
    timestamp: float | None = None


class HealthKitImportResponse(BaseModel):
    imported_count: int
    health: WatchHealthData
    samples: list[HealthKitSample]


class HealthKitSamplesResponse(BaseModel):
    samples: list[HealthKitSample]


class TrainingLoadResult(BaseModel):
    training_load: Literal["low", "medium", "high"]
    suggestion: Literal["continue", "reduce_intensity", "rest"]
    reason: str


class PoseAlgorithmRequest(BaseModel):
    keypoints: dict[str, RawKeypoint]
    exercise: Literal["squat"] = "squat"
    timestamp: float | None = None
    frame_index: int | None = None
    age: int = Field(default=20, ge=1, le=120)
    heart_rate: int | None = Field(default=None, ge=30, le=240)
    training_context: TrainingContext = Field(default_factory=TrainingContext)

    @model_validator(mode="before")
    @classmethod
    def parse_flat_yolo_pose(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "keypoints" in data:
            return data

        known_suffixes = (
            "shoulder",
            "elbow",
            "wrist",
            "hip",
            "knee",
            "ankle",
            "eye",
            "ear",
        )
        global_confidence = data.get("confidence", 1.0)
        keypoints: dict[str, Any] = {}
        remaining = dict(data)

        for name, value in data.items():
            is_keypoint = name.startswith(("left_", "right_")) and name.endswith(known_suffixes)
            if not is_keypoint:
                continue
            if isinstance(value, (list, tuple)) and len(value) == 2:
                keypoints[name] = [value[0], value[1], global_confidence]
            else:
                keypoints[name] = value
            remaining.pop(name, None)

        remaining.pop("confidence", None)
        remaining["keypoints"] = keypoints
        return remaining


class PoseAlgorithmResult(BaseModel):
    exercise: Literal["squat"]
    stage: Literal["standing", "descending", "bottom", "ascending", "unknown"]
    rep_count: int
    completed_rep: bool
    cleaning: PoseCleaningResult
    angles: JointAngles
    quality: QualityAssessment
    heart_rate_safety: HeartRateSafety | None = None
    training_load: TrainingLoadResult
    algorithm_context: str


class PoseAlgorithmResponse(BaseModel):
    result: PoseAlgorithmResult


class TrainingReportRequest(BaseModel):
    result: PoseAlgorithmResult
    goal: str | None = None


class TrainingReportResponse(BaseModel):
    report: str
    result: PoseAlgorithmResult


class PipelineAnalyzeRequest(BaseModel):
    user_id: str = "default"
    session_id: str | None = None
    keypoints: dict[str, RawKeypoint]
    exercise: Literal["squat"] = "squat"
    timestamp: float | None = None
    frame_index: int | None = None
    watch: WatchHealthData | None = None
    training_context: TrainingContext = Field(default_factory=TrainingContext)
    goal: str | None = None
    include_report: bool = True

    @model_validator(mode="before")
    @classmethod
    def parse_flat_pipeline_pose(cls, data: Any) -> Any:
        return PoseAlgorithmRequest.parse_flat_yolo_pose(data)


class PipelineAnalyzeResponse(BaseModel):
    result: PoseAlgorithmResult
    report: str | None = None
    scene: SceneObservation | None = None
    watch: WatchHealthData | None = None
