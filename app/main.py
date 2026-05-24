from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from app.algorithm_service import pose_algorithm_engine
from app.coaching_service import build_frontend_effect, forward_to_voice_agent
from app.health_service import (
    import_healthkit_samples,
    ingest_shortcut_heart_rate,
    ingest_watch_health,
    latest_healthkit_samples,
    latest_watch_health,
)
from app.llm_service import ask_llm, build_training_report, generate_realtime_commentary
from app.schemas import (
    AskRequest,
    AskResponse,
    DetectResponse,
    HealthKitImportRequest,
    HealthKitImportResponse,
    HealthKitSamplesResponse,
    HealthResponse,
    PipelineAnalyzeRequest,
    PipelineAnalyzeResponse,
    PoseAlgorithmRequest,
    PoseAlgorithmResponse,
    RealtimeCoachRequest,
    RealtimeCoachResponse,
    ShortcutHeartRateRequest,
    ShortcutHeartRateResponse,
    TrainingContext,
    TrainingReportRequest,
    TrainingReportResponse,
    WatchHealthData,
    WatchHealthResponse,
    YoloSceneIngestRequest,
    YoloSceneIngestResponse,
)
from app.vision_service import (
    analyze_image,
    capture_camera_frame,
    get_model,
    get_model_path,
    ingest_yolo_scene,
    latest_scene,
    model_loaded,
    read_image_bytes,
)

app = FastAPI(
    title="AI Sports Assistant YOLO Backend",
    description="Convert YOLO detections into LLM-friendly structured scene observations.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    path = get_model_path()
    return HealthResponse(
        ok=path.exists(),
        model_loaded=model_loaded(),
        model_path=str(path),
        details={"model_file_exists": path.exists()},
    )


@app.post("/warmup", response_model=HealthResponse)
def warmup() -> HealthResponse:
    try:
        get_model()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return health()


@app.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...)) -> DetectResponse:
    try:
        image = read_image_bytes(await file.read())
        scene = analyze_image(image, source=file.filename or "uploaded_image")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return DetectResponse(scene=scene)


@app.get("/camera/scene", response_model=DetectResponse)
def camera_scene() -> DetectResponse:
    try:
        frame = capture_camera_frame()
        scene = analyze_image(frame, source="camera")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return DetectResponse(scene=scene)


@app.get("/scene", response_model=DetectResponse)
def scene() -> DetectResponse:
    current = latest_scene()
    if current is None:
        raise HTTPException(status_code=404, detail="No scene has been captured yet.")
    return DetectResponse(scene=current)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    current = latest_scene()
    answer = ask_llm(request.question, current)
    return AskResponse(answer=answer, scene=current if request.include_scene_json else None)


@app.post("/ingest/yolo/scene", response_model=YoloSceneIngestResponse)
def ingest_external_yolo_scene(request: YoloSceneIngestRequest) -> YoloSceneIngestResponse:
    try:
        scene_observation = ingest_yolo_scene(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return YoloSceneIngestResponse(scene=scene_observation)


@app.post("/ingest/watch", response_model=WatchHealthResponse)
def ingest_watch(data: WatchHealthData) -> WatchHealthResponse:
    return WatchHealthResponse(health=ingest_watch_health(data))


@app.post("/ingest/shortcut/heart-rate", response_model=ShortcutHeartRateResponse)
def ingest_shortcut_heart_rate_sample(request: ShortcutHeartRateRequest) -> ShortcutHeartRateResponse:
    try:
        return ingest_shortcut_heart_rate(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ingest/healthkit", response_model=HealthKitImportResponse)
def ingest_healthkit(request: HealthKitImportRequest) -> HealthKitImportResponse:
    try:
        return import_healthkit_samples(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/watch/latest", response_model=WatchHealthResponse)
def watch_latest(user_id: str = "default") -> WatchHealthResponse:
    return WatchHealthResponse(health=latest_watch_health(user_id=user_id))


@app.get("/watch/samples", response_model=HealthKitSamplesResponse)
def watch_samples(
    limit: int = Query(default=50, ge=1, le=500),
    sample_type: str | None = None,
    user_id: str = "default",
    session_id: str | None = None,
) -> HealthKitSamplesResponse:
    return HealthKitSamplesResponse(
        samples=latest_healthkit_samples(
            limit=limit,
            sample_type=sample_type,
            user_id=user_id,
            session_id=session_id,
        )
    )


@app.post("/pose/analyze", response_model=PoseAlgorithmResponse)
def analyze_pose(request: PoseAlgorithmRequest) -> PoseAlgorithmResponse:
    try:
        result = pose_algorithm_engine.analyze(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return PoseAlgorithmResponse(result=result)


@app.post("/pose/reset")
def reset_pose_state() -> dict[str, str]:
    pose_algorithm_engine.reset()
    return {"status": "reset"}


@app.post("/training/report", response_model=TrainingReportResponse)
def training_report(request: TrainingReportRequest) -> TrainingReportResponse:
    try:
        report = build_training_report(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return TrainingReportResponse(
        report=report,
        result=request.result,
        results=request.results,
        snapshots=request.snapshots,
    )


@app.post("/llm/realtime-coach", response_model=RealtimeCoachResponse)
def realtime_coach(request: RealtimeCoachRequest) -> RealtimeCoachResponse:
    try:
        commentary = generate_realtime_commentary(request)
        effect = build_frontend_effect(request.result)
        voice_forwarded = False
        voice_response = None
        if request.send_to_voice_agent:
            voice_forwarded, voice_response = forward_to_voice_agent(
                commentary,
                user_id=request.user_id,
                session_id=request.session_id,
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RealtimeCoachResponse(
        commentary=commentary,
        frontend_effect=effect,
        voice_forwarded=voice_forwarded,
        voice_response=voice_response,
        result=request.result,
    )


@app.post("/pipeline/analyze", response_model=PipelineAnalyzeResponse)
def pipeline_analyze(request: PipelineAnalyzeRequest) -> PipelineAnalyzeResponse:
    if request.watch:
        watch_data = request.watch.model_copy(
            update={
                "user_id": request.user_id,
                "session_id": request.watch.session_id or request.session_id,
            }
        )
        watch = ingest_watch_health(watch_data)
    else:
        watch = latest_watch_health(request.user_id)
    training_context = _merge_training_context(request.training_context, watch)

    algorithm_request = PoseAlgorithmRequest(
        keypoints=request.keypoints,
        pose_candidates=request.pose_candidates,
        exercise=request.exercise,
        timestamp=request.timestamp,
        frame_index=request.frame_index,
        age=watch.age if watch else 20,
        heart_rate=watch.heart_rate if watch else None,
        training_context=training_context,
    )

    try:
        result = pose_algorithm_engine.analyze(algorithm_request)
        report = build_training_report(result, request.goal) if request.include_report else None
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PipelineAnalyzeResponse(
        result=result,
        report=report,
        scene=latest_scene(),
        watch=watch,
    )


def _merge_training_context(
    training_context: TrainingContext,
    watch: WatchHealthData | None,
) -> TrainingContext:
    if watch is None:
        return training_context

    update = {"sleep_quality": watch.sleep_quality}
    if watch.heart_rate_recovery_seconds is not None:
        update["heart_rate_recovery_seconds"] = watch.heart_rate_recovery_seconds
    return training_context.model_copy(update=update)
