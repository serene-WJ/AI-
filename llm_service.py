from __future__ import annotations

import json

from openai import OpenAI

from app.config import settings
from app.schemas import PoseAlgorithmResult, SceneObservation

SYSTEM_PROMPT = """You are an AI sports assistant connected to a YOLO perception backend.
You receive structured scene observations. Use object labels, normalized coordinates,
positions, and confidence scores to answer directly. If the scene is insufficient,
say what visual information is missing."""

TRAINING_REPORT_PROMPT = """You are an AI sports coach. You receive rule-based algorithm
results, not raw guesses. Explain the result in Chinese, give practical training advice,
and produce a concise training report. Do not change rep counts or safety decisions."""


def ask_llm(question: str, scene: SceneObservation | None) -> str:
    if not settings.llm_api_key or settings.llm_api_key == "your-api-key-here":
        return "LLM_API_KEY is not configured. The backend can still return YOLO scene data."

    client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    scene_text = scene.llm_context if scene else "No scene has been captured yet."
    response = client.chat.completions.create(
        model=settings.llm_model,
        temperature=0.4,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{scene_text}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content.strip()


def build_training_report(result: PoseAlgorithmResult, goal: str | None = None) -> str:
    if not settings.llm_api_key or settings.llm_api_key == "your-api-key-here":
        return "LLM_API_KEY is not configured. The backend can still return algorithm results."

    client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    result_json = json.dumps(result.model_dump(), ensure_ascii=False, indent=2)
    goal_text = goal or "根据当前动作质量、安全状态和训练容量给出建议。"
    response = client.chat.completions.create(
        model=settings.llm_model,
        temperature=0.3,
        messages=[
            {"role": "system", "content": TRAINING_REPORT_PROMPT},
            {
                "role": "user",
                "content": f"训练目标：{goal_text}\n\n算法结果：\n{result_json}",
            },
        ],
    )
    return response.choices[0].message.content.strip()
