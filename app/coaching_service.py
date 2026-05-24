from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from app.config import settings
from app.schemas import FrontendEffect, PoseAlgorithmResult


def build_frontend_effect(result: PoseAlgorithmResult) -> FrontendEffect:
    score = result.quality.quality_score
    if score >= 100:
        return FrontendEffect(
            name="perfect",
            trigger=True,
            message="perfect",
            sound="perfect",
            min_score=100,
        )
    if score >= 90:
        return FrontendEffect(
            name="excellent",
            trigger=True,
            message="excellent",
            sound="excellent",
            min_score=90,
        )
    if score >= 80:
        return FrontendEffect(
            name="good",
            trigger=True,
            message="good",
            sound="good",
            min_score=80,
        )
    return FrontendEffect(name="none", trigger=False)


def forward_to_voice_agent(text: str, user_id: str, session_id: str | None) -> tuple[bool, dict[str, Any] | None]:
    if not settings.voice_agent_url:
        return False, {"skipped": "VOICE_AGENT_URL is not configured."}

    payload = {
        "messages": [
            {
                "role": "user",
                "content": text,
            }
        ],
        "metadata": {
            "user_id": user_id,
            "session_id": session_id,
            "source": "ai_sports_assistant",
        },
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if settings.voice_agent_api_key:
        headers["Authorization"] = f"Bearer {settings.voice_agent_api_key}"

    request = urllib.request.Request(
        settings.voice_agent_url,
        data=data,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=settings.voice_agent_timeout) as response:
            body = response.read().decode("utf-8")
            return True, _parse_response_body(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return False, {"status": exc.code, "body": body}
    except Exception as exc:
        return False, {"error": str(exc)}


def _parse_response_body(body: str) -> dict[str, Any]:
    if not body:
        return {}
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return {"body": body}
    return parsed if isinstance(parsed, dict) else {"body": parsed}
