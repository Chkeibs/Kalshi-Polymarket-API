from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.services.pipeline import CorePipelineService
from api.services.scanner import LiveScanner


pipeline_service = CorePipelineService()
scanner = LiveScanner()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    if scanner.running:
        await scanner.stop()


app = FastAPI(title="Arbitrage API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/pipeline/refresh")
def refresh_pipeline(batch_size: int = 40) -> dict[str, object]:
    try:
        return pipeline_service.refresh(batch_size=batch_size)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/scanner/start")
async def start_scanner() -> dict[str, object]:
    try:
        return await scanner.start()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/scanner/stop")
async def stop_scanner() -> dict[str, object]:
    return await scanner.stop()


@app.get("/scanner/status")
def scanner_status() -> dict[str, object]:
    return scanner.status()


@app.get("/opportunities")
def opportunities(min_profit: float = 0.0, min_roi: float = 0.0) -> dict[str, object]:
    return {"items": scanner.list_opportunities(min_profit=min_profit, min_roi=min_roi)}

