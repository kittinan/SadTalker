import os
import queue
import shutil
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from src.sadtalker_service import SadTalkerService


UTC = timezone.utc


def utcnow() -> datetime:
    return datetime.now(UTC)


def isoformat(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


@dataclass
class JobRecord:
    job_id: str
    status: str
    created_at: datetime
    options: Dict[str, object]
    work_dir: str
    source_path: Optional[str] = None
    audio_path: Optional[str] = None
    output_path: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    def to_response(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["created_at"] = isoformat(self.created_at)
        payload["started_at"] = isoformat(self.started_at)
        payload["finished_at"] = isoformat(self.finished_at)
        payload["download_url"] = (
            f"/jobs/{self.job_id}/file" if self.status == "completed" and self.output_path else None
        )
        return payload


@dataclass
class AppState:
    service: SadTalkerService
    results_dir: Path
    retention_hours: int
    job_queue: queue.Queue = field(default_factory=queue.Queue)
    jobs: Dict[str, JobRecord] = field(default_factory=dict)
    jobs_lock: threading.Lock = field(default_factory=threading.Lock)
    stop_event: threading.Event = field(default_factory=threading.Event)
    worker_thread: Optional[threading.Thread] = None
    host: str = "127.0.0.1"
    port: int = 8000


def create_app() -> FastAPI:
    checkpoint_dir = os.environ.get("SADTALKER_CHECKPOINT_DIR", "./checkpoints")
    results_dir = Path(os.environ.get("SADTALKER_RESULTS_DIR", "./results/api")).resolve()
    retention_hours = int(os.environ.get("SADTALKER_RETENTION_HOURS", "24"))
    host = os.environ.get("SADTALKER_HOST", "127.0.0.1")
    port = int(os.environ.get("SADTALKER_PORT", "8000"))

    state = AppState(
        service=SadTalkerService(checkpoint_path=checkpoint_dir, config_path="src/config"),
        results_dir=results_dir,
        retention_hours=retention_hours,
        host=host,
        port=port,
    )

    app = FastAPI(title="SadTalker API", version="1.0.0")
    app.state.runtime = state

    @app.on_event("startup")
    def startup() -> None:
        state.results_dir.mkdir(parents=True, exist_ok=True)
        cleanup_old_jobs(state)
        state.worker_thread = threading.Thread(target=worker_loop, args=(state,), daemon=True)
        state.worker_thread.start()

    @app.on_event("shutdown")
    def shutdown() -> None:
        state.stop_event.set()
        state.job_queue.put(None)
        if state.worker_thread:
            state.worker_thread.join(timeout=5)

    @app.get("/healthz")
    def healthz():
        checkpoint_status = state.service.get_checkpoint_status()
        payload = {
            "ready": checkpoint_status["ready"],
            "device": checkpoint_status["device"],
            "checkpoint_dir": checkpoint_status["checkpoint_dir"],
            "supported_sizes": checkpoint_status["supported_sizes"],
            "sizes": checkpoint_status["sizes"],
            "queue_depth": state.job_queue.qsize(),
        }
        return payload

    @app.post("/jobs", status_code=202)
    async def create_job(
        source_image: UploadFile = File(...),
        driven_audio: UploadFile = File(...),
        preprocess: Literal["crop", "resize", "full", "extcrop", "extfull"] = Form("crop"),
        still_mode: bool = Form(False),
        enhancer: Literal["none", "gfpgan"] = Form("none"),
        batch_size: int = Form(2),
        size: int = Form(256),
        pose_style: int = Form(0),
    ):
        if batch_size < 1:
            raise HTTPException(status_code=422, detail="batch_size must be >= 1")
        if size not in (256, 512):
            raise HTTPException(status_code=422, detail="size must be 256 or 512")
        if pose_style < 0 or pose_style >= 46:
            raise HTTPException(status_code=422, detail="pose_style must be between 0 and 45")

        checkpoint_status = state.service.get_checkpoint_status()
        size_status = checkpoint_status["sizes"][str(size)]
        if not size_status["ready"]:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": f"Required checkpoints for size {size} are missing",
                    "missing": size_status["missing"],
                },
            )

        job_id = str(uuid.uuid4())
        work_dir = state.results_dir / job_id
        upload_dir = work_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        source_path = await save_upload(upload_dir, source_image, "source_image")
        audio_path = await save_upload(upload_dir, driven_audio, "driven_audio")

        record = JobRecord(
            job_id=job_id,
            status="queued",
            created_at=utcnow(),
            options={
                "preprocess": preprocess,
                "still_mode": still_mode,
                "enhancer": enhancer,
                "batch_size": batch_size,
                "size": size,
                "pose_style": pose_style,
            },
            work_dir=str(work_dir),
            source_path=str(source_path),
            audio_path=str(audio_path),
        )
        with state.jobs_lock:
            state.jobs[job_id] = record

        cleanup_old_jobs(state)
        state.job_queue.put(job_id)
        return record.to_response()

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str):
        record = get_job_record(state, job_id)
        return record.to_response()

    @app.get("/jobs/{job_id}/file")
    def get_job_file(job_id: str):
        record = get_job_record(state, job_id)
        if record.status != "completed" or not record.output_path:
            raise HTTPException(status_code=409, detail=f"Job {job_id} is not completed")

        output_path = Path(record.output_path)
        if not output_path.exists():
            raise HTTPException(status_code=404, detail="Generated file no longer exists")

        return FileResponse(path=output_path, media_type="video/mp4", filename=output_path.name)

    return app


def get_job_record(state: AppState, job_id: str) -> JobRecord:
    with state.jobs_lock:
        record = state.jobs.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return record


async def save_upload(upload_dir: Path, upload: UploadFile, field_name: str) -> Path:
    filename = upload.filename or f"{field_name}.bin"
    destination = upload_dir / f"{field_name}_{Path(filename).name}"

    with destination.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)

    await upload.close()
    return destination


def worker_loop(state: AppState) -> None:
    while not state.stop_event.is_set():
        job_id = state.job_queue.get()
        if job_id is None:
            state.job_queue.task_done()
            continue

        with state.jobs_lock:
            record = state.jobs.get(job_id)
            if record is None:
                state.job_queue.task_done()
                continue
            record.status = "running"
            record.started_at = utcnow()

        try:
            output_path = state.service.run_job(
                source_image=record.source_path,
                driven_audio=record.audio_path,
                preprocess=record.options["preprocess"],
                still_mode=record.options["still_mode"],
                enhancer=None if record.options["enhancer"] == "none" else record.options["enhancer"],
                batch_size=record.options["batch_size"],
                size=record.options["size"],
                pose_style=record.options["pose_style"],
                result_dir=str(state.results_dir),
                job_id=record.job_id,
            )

            with state.jobs_lock:
                record.status = "completed"
                record.output_path = str(Path(output_path).resolve())
                record.finished_at = utcnow()
        except Exception as exc:
            with state.jobs_lock:
                record.status = "failed"
                record.error = str(exc)
                record.finished_at = utcnow()
        finally:
            cleanup_old_jobs(state)
            state.job_queue.task_done()


def cleanup_old_jobs(state: AppState) -> None:
    cutoff = utcnow() - timedelta(hours=state.retention_hours)
    stale_records = []

    with state.jobs_lock:
        for job_id, record in list(state.jobs.items()):
            if record.status not in {"completed", "failed"}:
                continue
            reference_time = record.finished_at or record.created_at
            if reference_time >= cutoff:
                continue
            stale_records.append((job_id, record.work_dir))
            del state.jobs[job_id]

    for _, work_dir in stale_records:
        shutil.rmtree(work_dir, ignore_errors=True)


app = create_app()
