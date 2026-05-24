from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from app.vision_service import analyze_image, capture_camera_frame, latest_scene, read_image_bytes

mcp = FastMCP("ai-sports-assistant-yolo")


@mcp.tool()
def detect_current_frame() -> dict:
    """Capture one camera frame and return a YOLO scene observation for LLM reasoning."""
    frame = capture_camera_frame()
    return analyze_image(frame, source="camera").model_dump()


@mcp.tool()
def get_latest_scene() -> dict:
    """Return the latest scene observation produced by the YOLO backend."""
    scene = latest_scene()
    if scene is None:
        return {"error": "No scene has been captured yet."}
    return scene.model_dump()


@mcp.tool()
def detect_image_file(path: str) -> dict:
    """Run YOLO on an image file path and return normalized objects and scene text."""
    with open(path, "rb") as handle:
        image = read_image_bytes(handle.read())
    return analyze_image(image, source=path).model_dump()


if __name__ == "__main__":
    mcp.run()
