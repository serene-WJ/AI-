from __future__ import annotations

import json

from openai import OpenAI

from app.config import settings
from app.schemas import (
    PoseAlgorithmResult,
    RealtimeCoachRequest,
    SceneObservation,
    TrainingReportRequest,
)

SYSTEM_PROMPT = """You are an AI sports assistant connected to a YOLO perception backend.
You receive structured scene observations. Use object labels, normalized coordinates,
positions, and confidence scores to answer directly. If the scene is insufficient,
say what visual information is missing."""

TRAINING_REPORT_PROMPT = """你是专业运动训练教练。你接收的是后端规则算法结果，
不是让你重新计数或重新判分。请用中文生成训练结束后的完整报告。

报告必须包含：
1. 训练概览：次数、评分、心率/训练负荷等。
2. 动作评分：解释分数来源，不要修改后端分数。
3. 专业动作拆解评价：按截图/关键帧逐条点评，每条必须包含截图引用、优势点评、尚待提升、提升建议。
4. 风险提醒：只基于算法和心率结果。
5. 夸奖激励：真诚、具体、不要空泛。

如果请求里只有截图引用而没有真实图片内容，你只能引用截图编号或 URL，不能假装看到了图片细节。"""

REALTIME_COACH_PROMPT = """你是一个运动陪练语音教练。你会收到后端算法的实时结果。
请输出一句中文短句，适合语音播报，长度不超过 24 个汉字。
风格可以有一点碎嘴吐槽，但必须安全、鼓励、具体。
不要输出 JSON，不要重新计数，不要改变分数。"""


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


def build_training_report(
    request_or_result: TrainingReportRequest | PoseAlgorithmResult,
    goal: str | None = None,
) -> str:
    if not settings.llm_api_key or settings.llm_api_key == "your-api-key-here":
        return "LLM_API_KEY is not configured. The backend can still return algorithm results."

    client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    if isinstance(request_or_result, PoseAlgorithmResult):
        request = TrainingReportRequest(result=request_or_result, goal=goal)
    else:
        request = request_or_result

    payload = request.model_dump()
    goal_text = request.goal or goal or "根据动作质量、安全状态和训练容量给出训练建议。"
    response = client.chat.completions.create(
        model=settings.llm_model,
        temperature=0.3,
        messages=[
            {"role": "system", "content": TRAINING_REPORT_PROMPT},
            {
                "role": "user",
                "content": (
                    f"训练目标：{goal_text}\n\n"
                    f"训练数据、算法结果和截图引用：\n"
                    f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
                ),
            },
        ],
    )
    return response.choices[0].message.content.strip()


def generate_realtime_commentary(request: RealtimeCoachRequest) -> str:
    if not settings.llm_api_key or settings.llm_api_key == "your-api-key-here":
        return _fallback_realtime_commentary(request.result)

    client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    payload = {
        "style": request.style,
        "algorithm_result": request.result.model_dump(),
        "watch": request.watch.model_dump() if request.watch else None,
        "scene_summary": request.scene.summary if request.scene else None,
    }
    response = client.chat.completions.create(
        model=settings.llm_model,
        temperature=0.7,
        messages=[
            {"role": "system", "content": REALTIME_COACH_PROMPT},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, indent=2),
            },
        ],
    )
    return response.choices[0].message.content.strip()


def _fallback_realtime_commentary(result: PoseAlgorithmResult) -> str:
    score = result.quality.quality_score
    if result.heart_rate_safety and result.heart_rate_safety.status == "stop":
        return "心率太高了，先停一下"
    if "knee_inward" in result.quality.errors:
        return "膝盖别往里扣，稳住"
    if "back_leaning_forward" in result.quality.errors:
        return "背别抢跑，胸口抬一点"
    if score >= 100:
        return "完美，这一下很漂亮"
    if score >= 90:
        return "优秀，继续保持节奏"
    if score >= 80:
        return "不错，再稳一点就更好"
    return "动作先放慢，质量优先"
