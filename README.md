# AI Sports Assistant Backend

这个后端负责把 YOLO 视觉数据和手表健康数据整理成统一格式，先交给规则算法分析，再把算法结果交给 DeepSeek 生成回答、运动建议和训练报告。
它已经复用了原 `yolo/agent-hackathon` 里的轻量模型 `yolo11n.pt`。

## 产品链路

```text
YOLO 识别视觉/姿态数据
  -> 后端接收并整理数据格式
  -> 规则算法计算阶段、次数、动作质量、心率安全和训练容量
  -> DeepSeek 分析算法结果
  -> 输出回答、运动建议、训练报告

Apple Watch / HealthKit 心率睡眠数据
  -> 后端直接接收
  -> 合并进规则算法
```

手表数据不需要过 YOLO。YOLO 只负责视觉相关的数据，比如目标框、人体关键点、场景物体。

## 功能

- FastAPI 后端：支持上传图片、摄像头抓拍、获取最近场景、向 LLM 提问。
- LLM 友好数据：归一化 bbox、中心点、空间位置、物体计数、运动场景提示、`llm_context`。
- 外部 YOLO 接收口：支持接收其他 YOLO 服务已经算好的检测框 JSON。
- 手表健康数据接收口：支持接收心率、睡眠质量、睡眠时长、心率恢复时间。
- 健康数据持久化：HealthKit 原始样本和最新健康摘要会保存到本地 SQLite。
- 姿态规则算法：支持关键点清洗、关节角度计算、深蹲阶段识别、后端状态机计数、动作质量评分、心率安全线、训练容量建议。
- Pipeline 总入口：把姿态数据、最新手表数据、算法结果、DeepSeek 报告串起来。
- DeepSeek 训练报告：把规则算法结果交给 LLM，总结运动建议和训练报告，LLM 不负责计数。
- 可选 MCP server：把 YOLO 能力暴露成 Agent 可调用工具。

## 运行

```powershell
cd C:\Users\yangs\AI_sports_assitant
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

打开：

- `http://127.0.0.1:8000/docs`
- `GET /health`
- `POST /detect`
- `GET /camera/scene`
- `GET /scene`
- `POST /ask`
- `POST /ingest/yolo/scene`
- `POST /ingest/watch`
- `POST /ingest/healthkit`
- `GET /watch/latest`
- `GET /watch/samples`
- `POST /pose/analyze`
- `POST /pose/reset`
- `POST /pipeline/analyze`
- `POST /training/report`

## 推荐联调顺序

### 1. 手表上传健康数据

后端不直接读取 Apple Watch。实际产品接入时，由 iOS App / Watch App 向用户申请 HealthKit 权限，读取心率和睡眠数据后，再把 JSON 发给后端。

这一步不是“临时跑通”，而是产品架构本身：Apple Watch 数据必须先经过 iPhone / Watch App 的授权读取，后端只能接收 App 上传的数据。仓库里的 `scripts/import_apple_health.py` 只是开发和验收工具，用来在没有 iOS App 时模拟 HealthKit 上传。

推荐使用批量导入接口：

```http
POST /ingest/healthkit
```

```json
{
  "user_id": "demo_user",
  "session_id": "squat_2026_05_23",
  "device_id": "apple_watch",
  "age": 20,
  "samples": [
    {
      "sample_type": "heart_rate",
      "value": 168,
      "unit": "bpm",
      "start_time": 1716400000,
      "end_time": 1716400005,
      "source": "apple_watch"
    },
    {
      "sample_type": "sleep",
      "value": 7.2,
      "unit": "hours",
      "start_time": 1716330000,
      "end_time": 1716355920,
      "source": "healthkit"
    },
    {
      "sample_type": "heart_rate_recovery",
      "value": 120,
      "unit": "seconds",
      "source": "apple_watch"
    }
  ]
}
```

后端会自动整理成算法需要的格式：

```json
{
  "health": {
    "user_id": "demo_user",
    "session_id": "squat_2026_05_23",
    "age": 20,
    "heart_rate": 168,
    "sleep_quality": "good",
    "sleep_hours": 7.2,
    "heart_rate_recovery_seconds": 120
  }
}
```

开发调试时也可以直接上传整理后的简化数据：

```http
POST /ingest/watch
```

```json
{
  "user_id": "demo_user",
  "session_id": "squat_2026_05_23",
  "age": 20,
  "heart_rate": 168,
  "sleep_quality": "fair",
  "sleep_hours": 7.2,
  "heart_rate_recovery_seconds": 120
}
```

查看最近一次整理后的手表数据：

```http
GET /watch/latest?user_id=demo_user
```

查看最近导入的 HealthKit 原始样本：

```http
GET /watch/samples?user_id=demo_user&limit=50
```

也可以按类型过滤：

```http
GET /watch/samples?user_id=demo_user&sample_type=heart_rate
```

### Apple 健康导出文件导入工具

如果暂时没有 iOS App，可以先用 iPhone 健康 App 导出数据，再用脚本把它导入后端。这是开发验证工具，不是最终产品接入方式。

1. 在 iPhone 打开“健康”App。
2. 点击右上角头像。
3. 选择“导出所有健康数据”。
4. 把生成的 `export.zip` 传到电脑。
5. 启动后端：

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

6. 运行导入脚本：

```powershell
python scripts\import_apple_health.py C:\path\to\export.zip --user-id demo_user --session-id squat_2026_05_23 --age 20 --backend http://127.0.0.1:8000
```

脚本会解析最近 7 天的心率和睡眠记录，调用 `/ingest/healthkit` 导入后端。

### 2. YOLO 上传普通视觉检测框

如果 YOLO 在别的服务里已经跑完，可以把检测框发到：

```http
POST /ingest/yolo/scene
```

```json
{
  "source": "external_yolo",
  "image_size": {"width": 640, "height": 480},
  "detections": [
    {
      "label": "person",
      "confidence": 0.91,
      "bbox": [100, 40, 420, 470],
      "normalized": false
    },
    {
      "label": "sports ball",
      "confidence": 0.83,
      "bbox": [430, 220, 500, 290],
      "normalized": false
    }
  ]
}
```

如果后端自己跑 YOLO，则继续用 `POST /detect` 上传图片。

### 3. YOLO Pose 上传姿态关键点并跑完整 pipeline

```http
POST /pipeline/analyze
```

```json
{
  "exercise": "squat",
  "include_report": true,
  "goal": "判断这次深蹲是否标准，并给出训练建议",
  "keypoints": {
    "left_shoulder": [320, 120, 0.92],
    "left_hip": [300, 260, 0.93],
    "left_knee": [310, 410, 0.9],
    "left_ankle": [315, 560, 0.91],
    "right_shoulder": [390, 122, 0.9],
    "right_hip": [405, 262, 0.91],
    "right_knee": [398, 412, 0.88],
    "right_ankle": [392, 558, 0.89]
  },
  "training_context": {
    "current_sets": 2,
    "quality_drop_count": 1,
    "duration_minutes": 25,
    "temperature_c": 26,
    "humidity": 60
  }
}
```

这个接口会自动使用最近一次 `/ingest/watch` 的心率和睡眠数据。如果请求里直接传 `watch` 字段，也会覆盖最近一次手表数据。

## 姿态算法接口

`POST /pose/analyze` 用于接收 YOLO pose 或其他姿态模型输出的关键点数据。算法层会先清洗关键点，再计算角度、识别深蹲阶段并更新计数状态机。

示例请求：

```json
{
  "exercise": "squat",
  "age": 20,
  "heart_rate": 168,
  "keypoints": {
    "left_shoulder": [320, 120, 0.92],
    "left_hip": [300, 260, 0.93],
    "left_knee": [310, 410, 0.9],
    "left_ankle": [315, 560, 0.91],
    "right_shoulder": [390, 122, 0.9],
    "right_hip": [405, 262, 0.91],
    "right_knee": [398, 412, 0.88],
    "right_ankle": [392, 558, 0.89]
  },
  "training_context": {
    "current_sets": 2,
    "quality_drop_count": 1,
    "heart_rate_recovery_seconds": 120,
    "sleep_quality": "fair",
    "duration_minutes": 25,
    "temperature_c": 26,
    "humidity": 60
  }
}
```

关键输出：

```json
{
  "result": {
    "stage": "descending",
    "rep_count": 3,
    "completed_rep": false,
    "angles": {
      "knee_angle": 142.3,
      "hip_angle": 96.8,
      "trunk_angle": 168.4,
      "trunk_forward_lean": 12.5
    },
    "quality": {
      "quality_score": 90,
      "errors": [],
      "warnings": ["movement_too_fast"]
    },
    "heart_rate_safety": {
      "status": "normal",
      "max_heart_rate": 200,
      "warning_line": 170,
      "stop_line": 180
    },
    "training_load": {
      "training_load": "low",
      "suggestion": "continue",
      "reason": "movement quality and safety signals are stable"
    }
  }
}
```

`POST /pose/reset` 会清空深蹲阶段和计数状态，适合新训练开始前调用。

`POST /training/report` 接收 `/pose/analyze` 返回的 `result`，调用 DeepSeek 生成中文训练建议和训练报告。

## 树莓派建议

使用 `yolo11n.pt`，把 `YOLO_IMAGE_SIZE` 保持在 `320` 或 `416`，`YOLO_CONFIDENCE` 可以先用 `0.35`。
模型是懒加载，第一次请求会慢一点，后面会复用同一个模型实例。

## 可选 MCP

```powershell
python -m app.mcp_server
```

Tools:

- `detect_current_frame`
- `get_latest_scene`
- `detect_image_file`

## 输出数据

LLM 最方便直接使用的是 `scene.llm_context`，精确结构化数据在 `scene.objects`。
针对运动助手，`scene.sports_hints` 会额外告诉 LLM 是否有人、是否检测到球类/球拍/滑板等器材：

```json
{
  "summary": "Detected 1 person. Key positions: person#1 is at center-middle, near.",
  "sports_hints": {
    "people_count": 1,
    "has_people": true,
    "equipment": ["sports ball"],
    "has_ball": true,
    "nearest_object": "person",
    "likely_sports_scene": true
  },
  "objects": [
    {
      "label": "person",
      "confidence": 0.91,
      "bbox": {"x1": 0.2, "y1": 0.1, "x2": 0.7, "y2": 0.95},
      "center": {"x": 0.45, "y": 0.52},
      "spatial": {
        "horizontal": "center",
        "vertical": "middle",
        "size": "large",
        "distance_hint": "near"
      }
    }
  ]
}
```
