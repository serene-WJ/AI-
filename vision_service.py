from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from app.config import settings
from app.schemas import (
    DetectionObject,
    ImageSize,
    NormalizedBox,
    NormalizedPoint,
    SceneObservation,
    SpatialRelation,
    YoloSceneIngestRequest,
)

_model: YOLO | None = None
_latest_scene: SceneObservation | None = None


def model_loaded() -> bool:
    return _model is not None


def get_model_path() -> Path:
    return settings.model_path


def get_model() -> YOLO:
    global _model
    if _model is None:
        if not settings.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {settings.model_path}")
        _model = YOLO(str(settings.model_path))
    return _model


def latest_scene() -> SceneObservation | None:
    return _latest_scene


def read_image_bytes(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Uploaded file is not a valid image.")
    return image


def capture_camera_frame() -> np.ndarray:
    cap = cv2.VideoCapture(settings.camera_id)
    try:
        ret, frame = cap.read()
    finally:
        cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"Camera {settings.camera_id} is not available.")
    return frame


def analyze_image(image: np.ndarray, source: str = "uploaded_image") -> SceneObservation:
    global _latest_scene

    model = get_model()
    height, width = image.shape[:2]
    result = model(
        image,
        conf=settings.confidence,
        imgsz=settings.image_size,
        verbose=False,
    )[0]

    detections: list[DetectionObject] = []
    for index, box in enumerate(result.boxes, start=1):
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        label = str(model.names.get(class_id, f"class_{class_id}"))

        normalized = _normalize_box(x1, y1, x2, y2, width, height)
        center = NormalizedPoint(
            x=round((normalized.x1 + normalized.x2) / 2, 4),
            y=round((normalized.y1 + normalized.y2) / 2, 4),
        )
        area_ratio = round(
            max(0.0, normalized.x2 - normalized.x1)
            * max(0.0, normalized.y2 - normalized.y1),
            4,
        )

        detections.append(
            DetectionObject(
                object_id=index,
                label=label,
                confidence=round(confidence, 4),
                bbox=normalized,
                center=center,
                area_ratio=area_ratio,
                spatial=_spatial_relation(center, area_ratio),
            )
        )

    counts = dict(Counter(item.label for item in detections))
    sports_hints = _build_sports_hints(counts, detections)
    summary = _build_summary(counts, detections)
    llm_context = _build_llm_context(summary, detections, sports_hints)

    _latest_scene = SceneObservation(
        timestamp=time.time(),
        source=source,
        image_size=ImageSize(width=width, height=height),
        object_count=len(detections),
        counts=counts,
        sports_hints=sports_hints,
        objects=detections,
        summary=summary,
        llm_context=llm_context,
    )
    return _latest_scene


def ingest_yolo_scene(request: YoloSceneIngestRequest) -> SceneObservation:
    global _latest_scene

    width = request.image_size.width
    height = request.image_size.height
    detections: list[DetectionObject] = []

    for index, item in enumerate(request.detections, start=1):
        x1, y1, x2, y2 = [float(value) for value in item.bbox]
        if item.normalized:
            normalized = NormalizedBox(
                x1=_clamp_normalized(x1),
                y1=_clamp_normalized(y1),
                x2=_clamp_normalized(x2),
                y2=_clamp_normalized(y2),
            )
        else:
            normalized = _normalize_box(x1, y1, x2, y2, width, height)

        center = NormalizedPoint(
            x=round((normalized.x1 + normalized.x2) / 2, 4),
            y=round((normalized.y1 + normalized.y2) / 2, 4),
        )
        area_ratio = round(
            max(0.0, normalized.x2 - normalized.x1)
            * max(0.0, normalized.y2 - normalized.y1),
            4,
        )
        detections.append(
            DetectionObject(
                object_id=item.object_id or index,
                label=item.label,
                confidence=round(item.confidence, 4),
                bbox=normalized,
                center=center,
                area_ratio=area_ratio,
                spatial=_spatial_relation(center, area_ratio),
            )
        )

    counts = dict(Counter(item.label for item in detections))
    sports_hints = _build_sports_hints(counts, detections)
    summary = _build_summary(counts, detections)
    llm_context = _build_llm_context(summary, detections, sports_hints)

    _latest_scene = SceneObservation(
        timestamp=request.timestamp or time.time(),
        source=request.source,
        image_size=request.image_size,
        object_count=len(detections),
        counts=counts,
        sports_hints=sports_hints,
        objects=detections,
        summary=summary,
        llm_context=llm_context,
    )
    return _latest_scene


def _normalize_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> NormalizedBox:
    return NormalizedBox(
        x1=_clamp_normalized(x1 / width),
        y1=_clamp_normalized(y1 / height),
        x2=_clamp_normalized(x2 / width),
        y2=_clamp_normalized(y2 / height),
    )


def _clamp_normalized(value: float) -> float:
    return round(min(1.0, max(0.0, value)), 4)


def _spatial_relation(center: NormalizedPoint, area_ratio: float) -> SpatialRelation:
    if center.x < 0.33:
        horizontal = "left"
    elif center.x > 0.66:
        horizontal = "right"
    else:
        horizontal = "center"

    if center.y < 0.33:
        vertical = "top"
    elif center.y > 0.66:
        vertical = "bottom"
    else:
        vertical = "middle"

    if area_ratio < 0.05:
        size = "small"
        distance_hint = "far"
    elif area_ratio > 0.2:
        size = "large"
        distance_hint = "near"
    else:
        size = "medium"
        distance_hint = "medium"

    return SpatialRelation(
        horizontal=horizontal,
        vertical=vertical,
        size=size,
        distance_hint=distance_hint,
    )


def _build_summary(counts: dict[str, int], detections: list[DetectionObject]) -> str:
    if not detections:
        return "No significant objects were detected in the scene."

    count_text = ", ".join(f"{count} {label}" for label, count in sorted(counts.items()))
    leading = sorted(detections, key=lambda item: item.confidence, reverse=True)[:3]
    position_text = "; ".join(
        f"{item.label}#{item.object_id} is at {item.spatial.horizontal}-{item.spatial.vertical}, "
        f"{item.spatial.distance_hint}"
        for item in leading
    )
    return f"Detected {count_text}. Key positions: {position_text}."


def _build_sports_hints(counts: dict[str, int], detections: list[DetectionObject]) -> dict[str, object]:
    equipment_labels = {
        "sports ball",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "skis",
        "snowboard",
        "surfboard",
        "tennis racket",
        "frisbee",
    }
    equipment = sorted(label for label in counts if label in equipment_labels)
    people = counts.get("person", 0)
    nearest = max(detections, key=lambda item: item.area_ratio, default=None)

    return {
        "people_count": people,
        "has_people": people > 0,
        "equipment": equipment,
        "has_ball": "sports ball" in counts,
        "nearest_object": nearest.label if nearest else None,
        "likely_sports_scene": people > 0 and bool(equipment),
    }


def _build_llm_context(
    summary: str,
    detections: list[DetectionObject],
    sports_hints: dict[str, object],
) -> str:
    lines = [
        "Scene observation from YOLO.",
        f"Summary: {summary}",
        f"Sports hints: {sports_hints}",
        "Coordinates are normalized from 0 to 1. Use spatial fields for natural reasoning.",
    ]
    for item in detections:
        lines.append(
            f"- {item.label}#{item.object_id}: conf={item.confidence}, "
            f"center=({item.center.x},{item.center.y}), "
            f"position={item.spatial.horizontal}-{item.spatial.vertical}, "
            f"size={item.spatial.size}, distance={item.spatial.distance_hint}"
        )
    return "\n".join(lines)
