from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


class Settings:
    model_path: Path = BASE_DIR / os.getenv("YOLO_MODEL_PATH", "models/yolo11n.pt")
    confidence: float = _env_float("YOLO_CONFIDENCE", 0.35)
    image_size: int = _env_int("YOLO_IMAGE_SIZE", 416)
    camera_id: int = _env_int("CAMERA_ID", 0)

    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "deepseek-chat")

    voice_agent_url: str = os.getenv("VOICE_AGENT_URL", "")
    voice_agent_api_key: str = os.getenv("VOICE_AGENT_API_KEY", "")
    voice_agent_timeout: int = _env_int("VOICE_AGENT_TIMEOUT", 5)


settings = Settings()
