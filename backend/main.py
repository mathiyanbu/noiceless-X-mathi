"""
SIH26052 NOICELESSX - FastAPI Control and Telemetry Backend.

Controls and observes the embedded C++ dual-mic real-time runtime over IPC.
Audio processing runs strictly in the C++ ALSA threads with SCHED_FIFO; audio samples
are never transmitted over HTTP/WebSocket.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.routers import health, system, audio, model, metrics, runtime
from backend.websocket import telemetry_ws


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Zero-mock observation and control backend for the NOICELESSX "
            "dual-microphone real-time speech enhancement system."
        ),
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Allow cross-origin requests from frontend dev servers (Vite, Next, React)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API Routers
    app.include_router(health.router)
    app.include_router(system.router)
    app.include_router(audio.router)
    app.include_router(model.router)
    app.include_router(metrics.router)
    app.include_router(runtime.router)

    # Register WebSocket Router
    app.include_router(telemetry_ws.router)

    import os
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    dist_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
    if os.path.isdir(dist_dir):
        assets_dir = os.path.join(dist_dir, "assets")
        if os.path.isdir(assets_dir):
            app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

        @app.get("/", include_in_schema=False)
        async def serve_index():
            return FileResponse(os.path.join(dist_dir, "index.html"))
    else:
        @app.get("/")
        async def root():
            return {
                "name": settings.app_name,
                "version": settings.app_version,
                "docs": "/docs",
                "status": "operational"
            }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.host, port=settings.port, reload=True)
