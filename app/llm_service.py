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

REALTIME_COACH_PROMPT = """你是一个幽默话痨、健身知识丰富的 AI 健身搭子。
你会收到后端算法的实时结果，请根据动作评分、错误、心率和训练阶段生成一句适合语音播报的中文吐槽建议。

输出规则：
- 只输出一句话，不要 JSON，不要重新计数，不要改变分数。
- 语音播报长度控制在 18 到 45 个汉字之间。
- 结构优先用“轻微吐槽 + 立刻建议”，例如“哎呀妈呀膝盖又想串门，往脚尖方向顶住。”
- 可以参考东北话、四川话、韩语口头感叹的表达习惯，如“哎呀妈呀”“莫慌”“稳到起”“아이고”，但不要整句外语。
- 安全干预优先级最高：如果 safety_intervention.action 是 stop_training，必须直接叫停训练，不要玩梗。
- 如果 safety_intervention.action 是 reduce_intensity，必须提醒降强度或短休，并给出对应动作的休息方式。
- 允许幽默比喻，但不能羞辱、恐吓、攻击身材或人格。
- 必须给出一个可执行动作建议：膝盖方向、核心、背部、节奏、幅度、呼吸或休息。
- 如果出现胸闷、胸痛、头晕、眼前发黑、明显气短，提醒立即停止并寻求现场帮助。
"""


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
        "safety_intervention": _build_safety_intervention(request.result),
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
        return f"现在停下训练，{_rest_guidance_for_exercise(result.exercise, 'stop')}"
    if result.heart_rate_safety and result.heart_rate_safety.status == "reduce_intensity":
        return f"心率有点冲，先降强度，{_rest_guidance_for_exercise(result.exercise, 'reduce_intensity')}"
    if "knee_inward" in result.quality.errors:
        return "哎呀妈呀膝盖又想串门，往脚尖方向顶住。"
    if "back_leaning_forward" in result.quality.errors:
        return "你这背有点抢戏哈，核心收紧，胸口抬一点。"
    if "insufficient_depth" in result.quality.errors:
        return "这下有点浅尝辄止，再稳稳往下坐一点。"
    if "movement_too_fast" in result.quality.errors:
        return "아이고 别开倍速，下降慢一拍，起来再发力。"
    if "left_right_unbalanced" in result.quality.errors:
        return "左右有点各练各的，重心放中间，脚底踩稳。"
    if "low_confidence" in result.quality.errors:
        return "镜头有点看不清你，站回画面中间再来一下。"
    if score >= 100:
        return "Perfect！这一下漂亮得很，节奏和控制都在线。"
    if score >= 90:
        return "Excellent！稳到起，保持这个节奏继续拿分。"
    if score >= 80:
        return "Good！动作有模有样，再把核心收紧一点。"
    return "动作先别急着帅，放慢半拍，质量优先。"


def _build_safety_intervention(result: PoseAlgorithmResult) -> dict[str, str]:
    status = result.heart_rate_safety.status if result.heart_rate_safety else "normal"
    if status == "stop":
        return {
            "action": "stop_training",
            "reason": result.heart_rate_safety.message if result.heart_rate_safety else "heart rate reached stop line",
            "rest_guidance": _rest_guidance_for_exercise(result.exercise, "stop"),
            "resume_condition": "心率回落到提醒线以下，呼吸平稳且没有头晕、胸闷、胸痛后，再考虑恢复低强度训练。",
        }
    if status == "reduce_intensity":
        return {
            "action": "reduce_intensity",
            "reason": result.heart_rate_safety.message if result.heart_rate_safety else "heart rate is high",
            "rest_guidance": _rest_guidance_for_exercise(result.exercise, "reduce_intensity"),
            "resume_condition": "心率下降、能完整说话且动作控制恢复后，再继续下一组。",
        }
    if result.training_load.suggestion == "rest":
        return {
            "action": "stop_training",
            "reason": result.training_load.reason,
            "rest_guidance": _rest_guidance_for_exercise(result.exercise, "stop"),
            "resume_condition": "训练负荷已经偏高，本轮优先结束或延长休息。",
        }
    return {
        "action": "continue",
        "reason": "heart rate and training load are currently acceptable",
        "rest_guidance": _rest_guidance_for_exercise(result.exercise, "normal"),
        "resume_condition": "继续保持当前节奏。",
    }


def _rest_guidance_for_exercise(exercise: str, intensity: str) -> str:
    if intensity == "stop":
        base = "不要立刻冲下一组，先做放松呼吸，若头晕胸闷就坐下并寻求帮助。"
    elif intensity == "reduce_intensity":
        base = "休息30到90秒，鼻吸口呼，让心率慢慢下来。"
    else:
        base = "组间保持轻松呼吸，下一组前确认动作能稳住。"

    guidance = {
        "squat": "深蹲先站稳或扶墙慢走，放松股四头肌和臀腿，别马上蹲第二组。",
        "lunge": "弓步蹲先并脚站稳，轻轻活动髋和小腿，确认膝盖不发软再继续。",
        "push_up": "俯卧撑先跪姿或婴儿式休息，放松肩、腕和胸，呼吸顺了再撑起来。",
        "plank": "平板支撑先跪下或侧卧休息，放松腹部和肩颈，别硬顶腰背。",
        "jumping_jack": "开合跳改成原地慢走，双手自然摆动，把呼吸和步频降下来。",
        "high_knees": "高抬腿改成原地慢走，脚跟踩实，等呼吸不顶了再继续。",
        "sit_up": "仰卧起坐先屈膝仰卧或侧躺，放松髋屈肌和腹部，别猛坐起。",
    }
    return f"{guidance.get(exercise, '先停止当前动作，慢走或坐下调整呼吸。')}{base}"
