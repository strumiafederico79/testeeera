from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from typing import Optional
import os, uuid, logging, time, asyncio
from mastering import (
    process_audio, analyze_audio, spectrum_analysis_fft, mix_advice,
    MASTERING_PRESETS, get_preset, PLATFORM_LOUDNESS_TARGETS, get_platform_target,
    process_audio_with_reference, _crop_preview, measure_lufs_integrated,
)
from streaming_engine import master_stream_to_pcm16
from stem_separation import separate_stems
from stem_analysis import analyze_stems_full
from system_monitor import get_system_stats
import ai_assistant
from pydantic import BaseModel
import librosa, numpy as np, soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Audio Mastering API", version="7.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR    = "uploads"
PROCESSED_DIR = "processed"
STEMS_DIR     = "processed_stems"   # subcarpeta por job_id con los 4 WAV de stems
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif"}
MAX_FILE_SIZE  = 200 * 1024 * 1024   # 200 MB
PROCESSED_TTL  = 3600

os.makedirs(UPLOAD_DIR,    exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(STEMS_DIR,     exist_ok=True)

jobs: dict = {}

def validate_audio_file(filename: str) -> None:
    if not filename or not isinstance(filename, str):
        raise HTTPException(400, "Nombre de archivo inválido o faltante.")
    ext = os.path.splitext(filename)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Formato '{ext}' no soportado. Válidos: {sorted(ALLOWED_EXTENSIONS)}")

def sanitize_track_name(name: Optional[str], fallback: str = "mastered") -> str:
    """Limpia un nombre de tema provisto por el usuario para usarlo como filename seguro."""
    if not name:
        return fallback
    name = name.strip()
    if not name:
        return fallback
    name = name.replace("/", "-").replace("\\", "-")
    name = "".join(c for c in name if c.isprintable())
    safe = "".join(c for c in name if c.isalnum() or c in " ._-()[]áéíóúÁÉÍÓÚñÑüÜ")
    safe = safe.strip(" .")
    safe = safe[:120]
    return safe or fallback

async def read_and_validate(file: UploadFile) -> bytes:
    data = await file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(413, f"Archivo demasiado grande. Máximo: {MAX_FILE_SIZE // 1024 // 1024} MB")
    return data

def _get_input_duration(input_path: str) -> Optional[float]:
    """Calcula la duración del archivo (sin decodificarlo entero) para que
    /dashboard pueda estimar el ETA del job (ver system_monitor.estimate_remaining).

    BUGFIX: esta clave nunca se seteaba en ningún lado del código, por lo que
    el ETA del dashboard en vivo siempre devolvía None / nunca calculaba nada.

    BUGFIX 2: la primera versión de este fix escribía "_input_duration_sec"
    directo en el dict `params` que también se usa para llamar a
    process_audio(**params) — como esa función no tiene **kwargs, tirar esa
    clave de más rompía el job con "unexpected keyword argument". Ahora esta
    función NO muta `params`; devuelve el valor y quien la llama lo guarda
    únicamente en jobs[job_id]["params"] (el dict de estado/dashboard, que
    nunca se le pasa a process_audio).
    """
    try:
        info = sf.info(input_path)
        if info.samplerate:
            return round(info.frames / info.samplerate, 3)
    except Exception as e:
        logger.warning(f"No se pudo calcular la duración de '{input_path}': {e}")
    return None


_BOOL_QUERY_KEYS = {
    "use_lufs_normalize", "comp_stereo_link", "mb_bypass", "mb_stereo_bypass",
    "use_stereo_enhancer", "glue_bypass",
}

def coerce_ws_chain_params(params: dict) -> dict:
    """Convierte params recibidos por WebSocket desde URLSearchParams/JSON.

    El frontend arma el stream preview a partir de URLSearchParams; eso convierte
    floats/bools a strings. FastAPI hace este casteo automáticamente en endpoints
    REST, pero el WebSocket no, y el DSP espera números/bools reales.
    """
    out = {}
    for key, value in params.items():
        if key in _BOOL_QUERY_KEYS:
            if isinstance(value, str):
                out[key] = value.strip().lower() in {"1", "true", "yes", "on", "sí", "si"}
            else:
                out[key] = bool(value)
            continue
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                continue
            try:
                out[key] = float(value)
                continue
            except ValueError:
                pass
        out[key] = value
    return out

def cleanup_old() -> None:
    now = time.time()
    try:
        for fname in os.listdir(PROCESSED_DIR):
            fpath = os.path.join(PROCESSED_DIR, fname)
            if os.path.isfile(fpath) and (now - os.path.getmtime(fpath)) > PROCESSED_TTL:
                os.remove(fpath)
    except Exception as e:
        logger.warning(f"cleanup error: {e}")
    try:
        import shutil
        for dirname in os.listdir(STEMS_DIR):
            dpath = os.path.join(STEMS_DIR, dirname)
            if os.path.isdir(dpath) and (now - os.path.getmtime(dpath)) > PROCESSED_TTL:
                shutil.rmtree(dpath, ignore_errors=True)
    except Exception as e:
        logger.warning(f"cleanup stems error: {e}")

def _make_progress_cb(job_id: str):
    """Crea el callback que process_audio()/process_audio_with_reference()
    invocan en cada etapa de la cadena. Actualiza directamente el dict
    `jobs[job_id]`, que ya es lo que devuelve GET /job/{id} — así el
    frontend puede pollear progreso/etapa sin ningún endpoint nuevo."""
    def _cb(pct: int, stage: str):
        job = jobs.get(job_id)
        if job is None:
            return
        job["progress"] = pct
        job["stage"] = stage
    return _cb

def run_mastering_job(job_id: str, input_path: str, params: dict):
    jobs[job_id]["status"] = "processing"
    jobs[job_id]["started_at"] = time.time()
    jobs[job_id]["progress"] = 0
    jobs[job_id]["stage"] = "Iniciando procesamiento"
    try:
        cleanup_old()
        result = process_audio(input_path, progress_cb=_make_progress_cb(job_id), **params)
        jobs[job_id].update(status="done", result=result, finished_at=time.time(),
                             progress=100, stage="Completado")
        logger.info(f"Job {job_id} done: {result['output_path']}")
    except Exception as e:
        jobs[job_id].update(status="error", error=str(e))
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

def run_reference_job(job_id: str, input_path: str, reference_path: str, params: dict):
    jobs[job_id]["status"] = "processing"
    jobs[job_id]["started_at"] = time.time()
    jobs[job_id]["progress"] = 0
    jobs[job_id]["stage"] = "Iniciando procesamiento"
    try:
        cleanup_old()
        result = process_audio_with_reference(
            input_path, reference_path, progress_cb=_make_progress_cb(job_id), **params
        )
        jobs[job_id].update(status="done", result=result, finished_at=time.time(),
                             progress=100, stage="Completado")
        logger.info(f"Job {job_id} (reference match) done: {result['output_path']}")
    except Exception as e:
        jobs[job_id].update(status="error", error=str(e))
        logger.error(f"Job {job_id} (reference match) failed: {e}", exc_info=True)
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(reference_path):
            os.remove(reference_path)

def run_stems_job(job_id: str, input_path: str):
    """Job de separación de stems (#13, Demucs). Mismo patrón que
    run_mastering_job/run_reference_job: actualiza jobs[job_id] in-place
    para que /job/{id} lo pueda pollear con progress/stage."""
    jobs[job_id]["status"] = "processing"
    jobs[job_id]["started_at"] = time.time()
    jobs[job_id]["progress"] = 0
    jobs[job_id]["stage"] = "Iniciando separación"
    try:
        cleanup_old()
        # BUGFIX potencial: librosa.load fuerza el mismo sr para todos los
        # canales y decodifica a float32; usamos el mismo loader que
        # /analyze y /spectrum para que el comportamiento con distintos
        # formatos (mp3/flac/etc.) sea consistente en toda la app.
        audio, sr = librosa.load(input_path, sr=None, mono=False)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        stems = separate_stems(audio, sr, progress_cb=_make_progress_cb(job_id))

        jobs[job_id]["stage"] = "Analizando stems"
        jobs[job_id]["progress"] = 96
        # Timeout duro: si el análisis se cuelga por cualquier motivo (ej. un
        # futuro conflicto de threads entre libs), el job termina en error
        # después de ANALYSIS_TIMEOUT_SEC en vez de quedar trabado para
        # siempre en 96% (que es justamente lo que pasó antes de este fix).
        import concurrent.futures
        ANALYSIS_TIMEOUT_SEC = 180
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(analyze_stems_full, stems, sr, measure_lufs_integrated)
        try:
            analysis = future.result(timeout=ANALYSIS_TIMEOUT_SEC)
            pool.shutdown(wait=False)
        except concurrent.futures.TimeoutError:
            # wait=False: si el thread realmente está colgado (deadlock),
            # esperar a que termine (shutdown default) nos colgaría acá
            # también. Lo abandonamos y seguimos.
            pool.shutdown(wait=False)
            raise RuntimeError(
                f"El análisis de stems no terminó en {ANALYSIS_TIMEOUT_SEC}s "
                f"(se colgó). Los stems separados están listos igual; "
                f"revisar stem_analysis.py."
            )

        stem_dir = os.path.join(STEMS_DIR, job_id)
        os.makedirs(stem_dir, exist_ok=True)
        stem_paths = {}
        for name, stem_audio in stems.items():
            out_path = os.path.join(stem_dir, f"{name}.wav")
            data_to_write = stem_audio.T if stem_audio.ndim == 2 else stem_audio
            sf.write(out_path, data_to_write, sr, subtype="PCM_24")
            stem_paths[name] = out_path

        jobs[job_id].update(
            status="done", finished_at=time.time(), progress=100, stage="Completado",
            stem_analysis=analysis, stem_paths=stem_paths,
            available_stems=list(stem_paths.keys()),
        )
        logger.info(f"Job {job_id} (stems) done: {list(stem_paths.keys())}")
    except Exception as e:
        jobs[job_id].update(status="error", error=str(e))
        logger.error(f"Job {job_id} (stems) failed: {e}", exc_info=True)
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {"service": "Audio Mastering API", "version": "7.0.0",
            "max_file_mb": MAX_FILE_SIZE // 1024 // 1024,
            "endpoints": ["/master", "/master/sync", "/master/reference", "/master/reference/sync",
                          "/preview", "/analyze", "/spectrum",
                          "/mix-advice", "/job/{id}", "/download/{id}", "/report/{id}",
                          "/stems/separate", "/stems/download/{id}/{stem}",
                          "/presets", "/preset/{name}", "/platform-targets",
                          "/dashboard", "/ws/dashboard", "/ws/master-stream"]}

@app.get("/health", tags=["Info"])
def health():
    return {"status": "ok", "jobs": len(jobs)}

@app.get("/presets", tags=["Presets"])
def list_presets():
    return {name: preset for name, preset in MASTERING_PRESETS.items()}

@app.get("/preset/{name}", tags=["Presets"])
def get_preset_endpoint(name: str):
    try:
        return get_preset(name)
    except KeyError as e:
        raise HTTPException(404, str(e))

@app.get("/platform-targets", tags=["Mastering"])
def platform_targets():
    return PLATFORM_LOUDNESS_TARGETS

@app.get("/dashboard", tags=["Dashboard"])
def dashboard():
    return get_system_stats(jobs)

@app.websocket("/ws/dashboard")
async def ws_dashboard(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(get_system_stats(jobs))
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"ws_dashboard error: {e}")

@app.post("/analyze", tags=["Análisis"])
async def analyze(file: UploadFile = File(...)):
    validate_audio_file(file.filename)
    data = await read_and_validate(file)
    tmp = os.path.join(UPLOAD_DIR, f"analyze_{uuid.uuid4().hex}")
    try:
        with open(tmp, "wb") as f: f.write(data)
        # BUGFIX: analyze_audio/librosa.load son CPU-bound y bloqueantes.
        # Llamarlos directo desde un `async def` congela todo el event loop
        # (incluído /ws/dashboard y cualquier otra request concurrente)
        # durante todo el análisis. Se corren en threadpool.
        audio, sr = await run_in_threadpool(librosa.load, tmp, sr=None, mono=False)
        if audio.ndim == 1: audio = audio[np.newaxis, :]
        result = await run_in_threadpool(analyze_audio, audio, sr)
        result["mix_advice"] = mix_advice(result)
        return result
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))
    finally:
        if os.path.exists(tmp): os.remove(tmp)

@app.post("/mix-advice", tags=["Análisis"])
async def get_mix_advice(file: UploadFile = File(...)):
    validate_audio_file(file.filename)
    data = await read_and_validate(file)
    tmp = os.path.join(UPLOAD_DIR, f"advice_{uuid.uuid4().hex}")
    try:
        with open(tmp, "wb") as f: f.write(data)
        # BUGFIX: mismo problema de bloqueo del event loop que en /analyze.
        audio, sr = await run_in_threadpool(librosa.load, tmp, sr=None, mono=False)
        if audio.ndim == 1: audio = audio[np.newaxis, :]
        analysis = await run_in_threadpool(analyze_audio, audio, sr)
        return {"analysis": analysis, **mix_advice(analysis)}
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))
    finally:
        if os.path.exists(tmp): os.remove(tmp)

class AiChatMessage(BaseModel):
    role: str
    content: str

class AiChatRequest(BaseModel):
    message: str
    history: list[AiChatMessage] = []
    analysis: Optional[dict] = None
    preset: Optional[str] = None
    platform: Optional[str] = None

@app.get("/ai/status", tags=["Asistente IA"])
async def ai_status():
    """Indica si el asistente de IA está configurado y listo para usarse."""
    available = ai_assistant.is_available()
    return {
        "available": available,
        "model": ai_assistant.AI_MODEL if available else None,
        "reason": None if available else ai_assistant.get_unavailable_reason(),
    }

@app.post("/ai/chat", tags=["Asistente IA"])
async def ai_chat(req: AiChatRequest):
    """Chat con el asistente de IA de mastering (estilo LANDR AI).

    El frontend manda el último análisis disponible (de /analyze, /mix-advice
    o el resultado de un job) para que las respuestas sean específicas al
    track del usuario, más el historial de la conversación actual.
    """
    if not req.message or not req.message.strip():
        raise HTTPException(400, "El mensaje no puede estar vacío.")
    try:
        result = await run_in_threadpool(
            ai_assistant.chat,
            req.message,
            [(m.model_dump() if hasattr(m, "model_dump") else m.dict()) for m in req.history],
            req.analysis,
            req.preset,
            req.platform,
        )
        # `result` es {"reply": str, "suggested_params": dict, "suggestion_summary": str|None}.
        # suggested_params viene vacío si el modelo no propuso ningún ajuste aplicable
        # (p.ej. preguntas teóricas) — el frontend solo muestra el botón de aplicar
        # cuando ese dict tiene contenido.
        return result
    except RuntimeError as e:
        # Asistente no configurado (falta API key, etc.)
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Error en /ai/chat: {e}", exc_info=True)
        raise HTTPException(500, "Error interno del asistente de IA.")

@app.post("/ai/auto-master", tags=["Asistente IA"])
async def ai_auto_master(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    output_format: str = Query("wav", pattern="^(wav|flac|mp3)$"),
    output_bit_depth: int = Query(24, description="Bit depth de salida (WAV/FLAC): 16, 24 o 32 (float). Se aplica dither TPDF si baja de 32."),
):
    """Mastering 100% automático (estilo LANDR AI): la IA analiza el track y
    calcula ella misma cada parámetro de la cadena de mastering (compresor, EQ,
    multibanda, estéreo, limiter, etc.) en base al análisis — ya no elige entre
    presets predefinidos, arma la combinación de parámetros a medida del track.
    Encola el job igual que los demás endpoints de mastering — se puede
    pollear con /job/{job_id} normalmente.
    """
    validate_audio_file(file.filename)
    data = await read_and_validate(file)
    job_id = uuid.uuid4().hex
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    with open(input_path, "wb") as f:
        f.write(data)

    try:
        # Analizamos el track para dárselo como contexto a la IA (mismo
        # análisis que usan /analyze y /mix-advice).
        audio, sr = await run_in_threadpool(librosa.load, input_path, sr=None, mono=False)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        analysis = await run_in_threadpool(analyze_audio, audio, sr)
        analysis["mix_advice"] = mix_advice(analysis)

        platform_options = list(PLATFORM_LOUDNESS_TARGETS.keys())

        decision = await run_in_threadpool(
            ai_assistant.decide_mastering, analysis, platform_options, audio, sr
        )
    except ValueError as e:
        if os.path.exists(input_path): os.remove(input_path)
        raise HTTPException(400, str(e))
    except Exception as e:
        if os.path.exists(input_path): os.remove(input_path)
        logger.error(f"Error en /ai/auto-master (análisis/decisión): {e}", exc_info=True)
        raise HTTPException(500, "No se pudo analizar el track para el mastering automático.")

    # `decision` ya trae TODOS los parámetros de la cadena, calculados por la
    # IA (o por la heurística de respaldo), listos para pasarle a process_audio.
    params = {k: v for k, v in decision.items() if k not in ("platform", "reasoning")}
    params["output_format"] = output_format
    params["output_bit_depth"] = output_bit_depth
    if decision.get("platform"):
        params["platform_target"] = decision["platform"]

    duration = _get_input_duration(input_path)
    job_params = {**params, "ai_decision": decision}
    if duration is not None:
        job_params["_input_duration_sec"] = duration

    jobs[job_id] = {
        "status": "queued", "filename": file.filename, "created_at": time.time(),
        "params": job_params, "ai_decision": decision, "ai_analysis": analysis,
        "progress": 0, "stage": "En cola",
    }
    background_tasks.add_task(run_mastering_job, job_id, input_path, params)
    logger.info(f"Auto-mastering IA: job {job_id} -> parámetros calculados por IA, platform={decision.get('platform')}")
    return {
        "job_id": job_id, "status": "queued",
        "ai_decision": decision, "analysis": analysis, "poll_url": f"/job/{job_id}",
    }

@app.post("/spectrum", tags=["Análisis"])
async def spectrum(
    file: UploadFile = File(...),
    n_fft: int = Query(4096, ge=256, le=16384),
    n_bins: int = Query(64, ge=8, le=256),
):
    validate_audio_file(file.filename)
    data = await read_and_validate(file)
    tmp = os.path.join(UPLOAD_DIR, f"spectrum_{uuid.uuid4().hex}")
    try:
        with open(tmp, "wb") as f: f.write(data)
        # BUGFIX: mismo problema de bloqueo del event loop que en /analyze.
        audio, sr = await run_in_threadpool(librosa.load, tmp, sr=None, mono=False)
        if audio.ndim == 1: audio = audio[np.newaxis, :]
        return await run_in_threadpool(spectrum_analysis_fft, audio, sr, n_fft=n_fft, n_bins=n_bins)
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))
    finally:
        if os.path.exists(tmp): os.remove(tmp)

@app.post("/stems/separate", tags=["Stems"])
async def stems_separate(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Separa el track en stems (vocals/drums/bass/other) con Demucs, analiza
    cada uno individualmente y detecta colisiones espectrales entre ellos
    (ej. kick tapando al bajo). Encola el job igual que /master — se pollea
    con el mismo /job/{job_id} de siempre."""
    validate_audio_file(file.filename)
    data = await read_and_validate(file)
    job_id = uuid.uuid4().hex
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    with open(input_path, "wb") as f:
        f.write(data)

    duration = _get_input_duration(input_path)
    job_params = {}
    if duration is not None:
        job_params["_input_duration_sec"] = duration

    jobs[job_id] = {
        "status": "queued", "type": "stems", "filename": file.filename,
        "created_at": time.time(), "params": job_params, "progress": 0, "stage": "En cola",
    }
    background_tasks.add_task(run_stems_job, job_id, input_path)
    return {"job_id": job_id, "status": "queued", "poll_url": f"/job/{job_id}"}


@app.get("/stems/download/{job_id}/{stem_name}", tags=["Stems"])
def stems_download(job_id: str, stem_name: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job no encontrado")
    job = jobs[job_id]
    if job.get("type") != "stems" or job["status"] != "done":
        raise HTTPException(400, f"Job no listo: {job.get('status')}")
    stem_path = job.get("stem_paths", {}).get(stem_name)
    if not stem_path or not os.path.exists(stem_path):
        raise HTTPException(410, "Stem no encontrado o expirado. Volvé a separar el track.")
    return FileResponse(stem_path, media_type="audio/wav", filename=f"{stem_name}.wav")


@app.websocket("/ws/master-stream")
async def ws_master_stream(websocket: WebSocket):
    await websocket.accept()
    tmp_path = None
    try:
        config_msg = await websocket.receive_json()
        chunk_seconds = float(config_msg.get("chunk_seconds", 2.0))
        preset_name = config_msg.get("preset")
        platform = config_msg.get("platform_target")
        # BUGFIX: el preview en vivo (streaming) nunca recortaba el audio, así
        # que procesaba y transmitía el archivo COMPLETO en cada cambio de
        # parámetro en vez del extracto corto que indica la UI ("backend 10s").
        # Se guarda acá para recortar el audio una vez cargado (ver más abajo),
        # con el mismo criterio que ya usa /preview: 10s por defecto.
        preview_seconds_stream = config_msg.get("preview_seconds")
        chain_params = {k: v for k, v in config_msg.items() if k not in ("chunk_seconds", "preset", "platform_target", "preview_seconds", "type")}

        if preset_name:
            chain_params = {**get_preset(preset_name), **chain_params}
            chain_params.pop("label", None)
        if platform:
            chain_params["use_lufs_normalize"] = True
            chain_params["target_lufs"] = get_platform_target(platform)["lufs"]

        chain_params = coerce_ws_chain_params(chain_params)

        audio_bytes = await websocket.receive_bytes()

        # BUGFIX: a diferencia de todos los endpoints REST (/master, /preview,
        # /analyze...), este websocket nunca validaba el tamaño del archivo
        # recibido. Cualquier cliente podía mandar un binario de cualquier
        # tamaño y agotar la memoria/disco del servidor. Se aplica el mismo
        # límite (MAX_FILE_SIZE) que usa read_and_validate() para las subidas
        # por HTTP.
        if len(audio_bytes) > MAX_FILE_SIZE:
            await websocket.send_json({
                "event": "error",
                "message": f"Archivo demasiado grande. Máximo: {MAX_FILE_SIZE // 1024 // 1024} MB",
            })
            return

        tmp_path = os.path.join(UPLOAD_DIR, f"stream_{uuid.uuid4().hex}")
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)

        # BUGFIX: librosa.load() y el procesamiento de cada chunk (filtros,
        # compresor, oversampling) son CPU-bound. Ejecutarlos directo en la
        # coroutine del websocket bloquea el event loop, lo que congela
        # /ws/dashboard y cualquier otra conexión/request mientras dura el
        # streaming. Se corren en threadpool: la carga del archivo entera,
        # y cada chunk se calcula en threadpool antes de mandarlo, cediendo
        # el control al event loop entre chunk y chunk.
        audio, sr = await run_in_threadpool(librosa.load, tmp_path, sr=None, mono=False)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        # BUGFIX: recortar acá (y no antes) para no gastar tiempo de I/O de más;
        # igual que en /preview, se toma un extracto centrado en la MITAD del
        # tema (el arranque suele ser intro/silencio poco representativo).
        # Con esto el preview en vivo es ~instantáneo sin importar si el tema
        # dura 10 segundos o 10 minutos.
        preview_window = float(preview_seconds_stream) if preview_seconds_stream else 10.0
        audio = _crop_preview(audio, sr, preview_window)

        chain_params.pop("output_format", None)
        chain_params.pop("preview_seconds", None)

        chunk_gen = master_stream_to_pcm16(audio, sr, chunk_seconds=chunk_seconds, **chain_params)
        _SENTINEL = object()
        while True:
            item = await run_in_threadpool(next, chunk_gen, _SENTINEL)
            if item is _SENTINEL:
                break
            pcm_bytes, metrics = item
            await websocket.send_json({"event": "chunk", "metrics": metrics, "sample_rate": sr, "channels": int(audio.shape[0])})
            await websocket.send_bytes(pcm_bytes)

        await websocket.send_json({"event": "done"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"ws_master_stream error: {e}", exc_info=True)
        try:
            await websocket.send_json({"event": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

# ─── Endpoint con preset (parámetros multibanda ahora opcionales) ──────────────────
@app.post("/master/preset/{preset_name}", tags=["Mastering"])
async def master_with_preset(
    preset_name: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    platform_target: str = Query(None, description="spotify|youtube|apple_music|tidal|club|cd"),
    output_format: str = Query("wav", pattern="^(wav|flac|mp3)$"),
    output_bit_depth: int = Query(24, description="Bit depth de salida (WAV/FLAC): 16, 24 o 32 (float). Se aplica dither TPDF si baja de 32."),
    # Parámetros multibanda opcionales (si no se envían, se respeta el preset)
    mb_low_crossover: float = Query(None, ge=20.0, le=2000.0),
    mb_high_crossover: float = Query(None, ge=500.0, le=20000.0),
    mb_low_threshold: float = Query(None, ge=0.0, le=1.0),
    mb_low_ratio: float = Query(None, ge=1.0, le=20.0),
    mb_low_attack_ms: float = Query(None, ge=0.1, le=200.0),
    mb_low_release_ms: float = Query(None, ge=10.0, le=1000.0),
    mb_low_makeup_db: float = Query(None, ge=-12.0, le=24.0),
    mb_mid_threshold: float = Query(None, ge=0.0, le=1.0),
    mb_mid_ratio: float = Query(None, ge=1.0, le=20.0),
    mb_mid_attack_ms: float = Query(None, ge=0.1, le=200.0),
    mb_mid_release_ms: float = Query(None, ge=10.0, le=1000.0),
    mb_mid_makeup_db: float = Query(None, ge=-12.0, le=24.0),
    mb_high_threshold: float = Query(None, ge=0.0, le=1.0),
    mb_high_ratio: float = Query(None, ge=1.0, le=20.0),
    mb_high_attack_ms: float = Query(None, ge=0.1, le=200.0),
    mb_high_release_ms: float = Query(None, ge=10.0, le=1000.0),
    mb_high_makeup_db: float = Query(None, ge=-12.0, le=24.0),
    mb_bypass: Optional[bool] = Query(None),
    input_gain_db: Optional[float] = Query(None, ge=-24.0, le=24.0),
):
    try:
        params = get_preset(preset_name)
    except KeyError as e:
        raise HTTPException(404, str(e))
    params.pop("label", None)
    params["output_format"] = output_format
    params["output_bit_depth"] = output_bit_depth
    if platform_target:
        params["platform_target"] = platform_target
    # Solo sobrescribir si el usuario envió el valor (no None)
    for key in ["mb_low_crossover", "mb_high_crossover", "mb_low_threshold", "mb_low_ratio",
                "mb_low_attack_ms", "mb_low_release_ms", "mb_low_makeup_db",
                "mb_mid_threshold", "mb_mid_ratio", "mb_mid_attack_ms", "mb_mid_release_ms",
                "mb_mid_makeup_db", "mb_high_threshold", "mb_high_ratio", "mb_high_attack_ms",
                "mb_high_release_ms", "mb_high_makeup_db"]:
        val = locals().get(key)
        if val is not None:
            params[key] = val
    if mb_bypass is not None:
        params["mb_bypass"] = mb_bypass
    if input_gain_db is not None:
        params["input_gain_db"] = input_gain_db

    validate_audio_file(file.filename)
    data = await read_and_validate(file)
    job_id = uuid.uuid4().hex
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    with open(input_path, "wb") as f: f.write(data)

    duration = _get_input_duration(input_path)
    job_params = {**params, "preset": preset_name}
    if duration is not None:
        job_params["_input_duration_sec"] = duration
    jobs[job_id] = {"status": "queued", "filename": file.filename, "created_at": time.time(),
                     "params": job_params, "progress": 0, "stage": "En cola"}
    background_tasks.add_task(run_mastering_job, job_id, input_path, params)
    return {"job_id": job_id, "status": "queued", "preset": preset_name, "poll_url": f"/job/{job_id}"}

# ── Preview ──────────────────────────────────────────────────────────────────
@app.post("/preview", tags=["Preview"])
async def preview(
    file: UploadFile = File(...),
    target_peak: float        = Query(0.95,   ge=0.1,   le=1.0),
    use_lufs_normalize: bool  = Query(False),
    target_lufs: float        = Query(-14.0,  ge=-40.0, le=0.0),
    # Compresor multibanda con valores más conservadores
    mb_low_crossover: float = Query(250.0, ge=20.0, le=2000.0),
    mb_high_crossover: float = Query(4000.0, ge=500.0, le=20000.0),
    mb_low_threshold: float = Query(0.7, ge=0.0, le=1.0),
    mb_low_ratio: float = Query(2.0, ge=1.0, le=20.0),
    mb_low_attack_ms: float = Query(20.0, ge=0.1, le=200.0),
    mb_low_release_ms: float = Query(150.0, ge=10.0, le=1000.0),
    mb_low_makeup_db: float = Query(0.0, ge=-12.0, le=24.0),
    mb_mid_threshold: float = Query(0.7, ge=0.0, le=1.0),
    mb_mid_ratio: float = Query(2.0, ge=1.0, le=20.0),
    mb_mid_attack_ms: float = Query(20.0, ge=0.1, le=200.0),
    mb_mid_release_ms: float = Query(150.0, ge=10.0, le=1000.0),
    mb_mid_makeup_db: float = Query(0.0, ge=-12.0, le=24.0),
    mb_high_threshold: float = Query(0.7, ge=0.0, le=1.0),
    mb_high_ratio: float = Query(2.0, ge=1.0, le=20.0),
    mb_high_attack_ms: float = Query(20.0, ge=0.1, le=200.0),
    mb_high_release_ms: float = Query(150.0, ge=10.0, le=1000.0),
    mb_high_makeup_db: float = Query(0.0, ge=-12.0, le=24.0),
    mb_bypass: bool = Query(True),
    input_gain_db: float      = Query(0.0,    ge=-24.0, le=24.0),
    comp_threshold: float      = Query(0.5,   ge=0.0,   le=1.0),
    comp_ratio: float          = Query(4.0,   ge=1.0,   le=20.0),
    comp_attack_ms: float      = Query(10.0,  ge=0.1,   le=200.0),
    comp_release_ms: float     = Query(100.0, ge=10.0,  le=1000.0),
    comp_makeup_db: float      = Query(0.0,   ge=-12.0, le=24.0),
    comp_stereo_link: bool     = Query(True, description="Linkea L/R en el compresor para preservar la imagen estéreo"),
    oversample_mode: str       = Query("quality", pattern="^(off|draft|fast|quality|ultra)$"),
    # EQ
    hp_cutoff: float          = Query(80.0,   ge=20.0,  le=500.0),
    high_shelf_gain_db: float = Query(2.0,    ge=-12.0, le=12.0),
    high_shelf_freq_hz: float  = Query(8000.0, ge=1000.0, le=20000.0),
    # Multiband Stereo Width
    mb_stereo_bypass: bool          = Query(True),
    mb_stereo_low_width: float      = Query(0.9,   ge=0.0, le=3.0),
    mb_stereo_mid_width: float      = Query(1.2,   ge=0.0, le=3.0),
    mb_stereo_high_width: float     = Query(1.5,   ge=0.0, le=3.0),
    mb_stereo_low_crossover: float  = Query(150.0, ge=20.0, le=2000.0),
    mb_stereo_high_crossover: float = Query(4000.0,ge=200.0, le=20000.0),
    eq1_freq: float = Query(100.0), eq1_gain: float = Query(0.0), eq1_q: float = Query(1.0),
    eq2_freq: float = Query(500.0), eq2_gain: float = Query(0.0), eq2_q: float = Query(1.0),
    eq3_freq: float = Query(2000.0), eq3_gain: float = Query(0.0), eq3_q: float = Query(1.0),
    eq4_freq: float = Query(8000.0), eq4_gain: float = Query(0.0), eq4_q: float = Query(1.0),
    transient_attack: float   = Query(0.0,   ge=-1.0,  le=1.0),
    transient_sustain: float  = Query(0.0,   ge=-1.0,  le=1.0),
    saturation_drive: float   = Query(0.0,   ge=0.0,   le=1.0),
    saturation_mode: str      = Query("tape", pattern="^(tape|tube)$"),
    saturation_mix: float     = Query(1.0,   ge=0.0,   le=1.0),
    mid_gain_db: float        = Query(0.0,   ge=-12.0, le=12.0),
    side_gain_db: float       = Query(0.0,   ge=-18.0, le=18.0),
    stereo_width_amount: float = Query(1.2,  ge=0.0,   le=3.0),
    use_stereo_enhancer: bool  = Query(False),
    enhancer_bass_mono_freq: float = Query(120.0),
    haas_delay_ms: float      = Query(0.0,   ge=0.0,   le=30.0),
    reverb_size: float        = Query(0.3,   ge=0.05,  le=2.0),
    reverb_wet: float         = Query(0.0,   ge=0.0,   le=1.0),
    glue_bypass: bool         = Query(True),
    glue_threshold_db: float  = Query(-4.0,  ge=-24.0, le=0.0),
    glue_ratio: float         = Query(2.0,   ge=1.0,   le=10.0),
    glue_attack_ms: float     = Query(30.0,  ge=0.1,   le=200.0),
    glue_release_ms: float    = Query(120.0, ge=10.0,  le=1000.0),
    glue_makeup_db: float     = Query(0.0,   ge=-12.0, le=12.0),
    limiter_ceiling: float    = Query(0.95,  ge=0.5,   le=1.0),
    limiter_release_ms: float = Query(50.0,  ge=1.0,   le=500.0),
    # EQ de fase lineal (FIR) / Dynamic EQ / Low-End Mono Maker dedicado
    eq_mode: str              = Query("iir", pattern="^(iir|linear_phase)$"),
    linear_phase_taps: int    = Query(2049, ge=257, le=8193),
    low_end_mono_freq: float  = Query(120.0, ge=40.0, le=300.0),
    low_end_mono_amount: float = Query(0.0, ge=0.0, le=1.0),
    dyneq_bypass: bool        = Query(True),
    dyneq_freq: float         = Query(3000.0, ge=200.0, le=16000.0),
    dyneq_q: float            = Query(2.5,   ge=0.5,  le=12.0),
    dyneq_threshold_db: float = Query(-18.0, ge=-60.0, le=0.0),
    dyneq_ratio: float        = Query(3.0,   ge=1.0,  le=20.0),
    dyneq_attack_ms: float    = Query(3.0,   ge=0.1,  le=100.0),
    dyneq_release_ms: float   = Query(80.0,  ge=5.0,  le=1000.0),
    dyneq_max_reduction_db: float = Query(12.0, ge=0.0, le=30.0),
    nr_bypass: bool           = Query(True,  description="Desactivar para aplicar reducción de ruido antes de la cadena."),
    nr_strength: float        = Query(0.5,   ge=0.0, le=1.0, description="Intensidad de la reducción de ruido (0=nada, 1=máximo)."),
    nr_noise_sample_sec: float = Query(0.5,  ge=0.1, le=5.0, description="Segundos iniciales usados para estimar el perfil de ruido."),
    output_format: str        = Query("mp3", pattern="^(wav|flac|mp3)$"),
    output_bit_depth: int     = Query(24, description="Bit depth de salida (WAV/FLAC): 16, 24 o 32 (float). Se aplica dither TPDF si baja de 32."),
    preview_seconds: float    = Query(10.0,  ge=5.0,   le=120.0),
    platform_target: str      = Query(None,  pattern="^(spotify|youtube|apple_music|tidal|club|cd)$"),
):
    validate_audio_file(file.filename)
    data = await read_and_validate(file)
    tmp = os.path.join(UPLOAD_DIR, f"prev_{uuid.uuid4().hex}")
    try:
        with open(tmp, "wb") as f: f.write(data)
        # BUGFIX: process_audio() es la función más pesada de toda la API
        # (filtros, compresor, oversampling x4, etc). Llamarla directo acá
        # bloqueaba el event loop durante TODO el preview, congelando el
        # dashboard en vivo y cualquier otra request mientras tanto.
        result = await run_in_threadpool(
            process_audio,
            tmp,
            target_peak=target_peak,
            use_lufs_normalize=use_lufs_normalize,
            target_lufs=target_lufs,
            input_gain_db=input_gain_db,
            oversample_mode=oversample_mode,
            comp_stereo_link=comp_stereo_link,
            comp_threshold=comp_threshold,
            comp_ratio=comp_ratio,
            comp_attack_ms=comp_attack_ms,
            comp_release_ms=comp_release_ms,
            comp_makeup_db=comp_makeup_db,
            mb_low_crossover=mb_low_crossover,
            mb_high_crossover=mb_high_crossover,
            mb_low_threshold=mb_low_threshold,
            mb_low_ratio=mb_low_ratio,
            mb_low_attack_ms=mb_low_attack_ms,
            mb_low_release_ms=mb_low_release_ms,
            mb_low_makeup_db=mb_low_makeup_db,
            mb_mid_threshold=mb_mid_threshold,
            mb_mid_ratio=mb_mid_ratio,
            mb_mid_attack_ms=mb_mid_attack_ms,
            mb_mid_release_ms=mb_mid_release_ms,
            mb_mid_makeup_db=mb_mid_makeup_db,
            mb_high_threshold=mb_high_threshold,
            mb_high_ratio=mb_high_ratio,
            mb_high_attack_ms=mb_high_attack_ms,
            mb_high_release_ms=mb_high_release_ms,
            mb_high_makeup_db=mb_high_makeup_db,
            mb_bypass=mb_bypass,
            hp_cutoff=hp_cutoff,
            high_shelf_gain_db=high_shelf_gain_db,
            high_shelf_freq_hz=high_shelf_freq_hz,
            mb_stereo_bypass=mb_stereo_bypass,
            mb_stereo_low_width=mb_stereo_low_width,
            mb_stereo_mid_width=mb_stereo_mid_width,
            mb_stereo_high_width=mb_stereo_high_width,
            mb_stereo_low_crossover=mb_stereo_low_crossover,
            mb_stereo_high_crossover=mb_stereo_high_crossover,
            eq1_freq=eq1_freq, eq1_gain=eq1_gain, eq1_q=eq1_q,
            eq2_freq=eq2_freq, eq2_gain=eq2_gain, eq2_q=eq2_q,
            eq3_freq=eq3_freq, eq3_gain=eq3_gain, eq3_q=eq3_q,
            eq4_freq=eq4_freq, eq4_gain=eq4_gain, eq4_q=eq4_q,
            transient_attack=transient_attack,
            transient_sustain=transient_sustain,
            saturation_drive=saturation_drive,
            saturation_mode=saturation_mode,
            saturation_mix=saturation_mix,
            mid_gain_db=mid_gain_db,
            side_gain_db=side_gain_db,
            stereo_width_amount=stereo_width_amount,
            use_stereo_enhancer=use_stereo_enhancer,
            enhancer_bass_mono_freq=enhancer_bass_mono_freq,
            haas_delay_ms=haas_delay_ms,
            reverb_size=reverb_size,
            reverb_wet=reverb_wet,
            glue_bypass=glue_bypass,
            glue_threshold_db=glue_threshold_db,
            glue_ratio=glue_ratio,
            glue_attack_ms=glue_attack_ms,
            glue_release_ms=glue_release_ms,
            glue_makeup_db=glue_makeup_db,
            limiter_ceiling=limiter_ceiling,
            limiter_release_ms=limiter_release_ms,
            eq_mode=eq_mode,
            linear_phase_taps=linear_phase_taps,
            low_end_mono_freq=low_end_mono_freq,
            low_end_mono_amount=low_end_mono_amount,
            dyneq_bypass=dyneq_bypass,
            dyneq_freq=dyneq_freq,
            dyneq_q=dyneq_q,
            dyneq_threshold_db=dyneq_threshold_db,
            dyneq_ratio=dyneq_ratio,
            dyneq_attack_ms=dyneq_attack_ms,
            dyneq_release_ms=dyneq_release_ms,
            dyneq_max_reduction_db=dyneq_max_reduction_db,
            nr_bypass=nr_bypass,
            nr_strength=nr_strength,
            nr_noise_sample_sec=nr_noise_sample_sec,
            output_format=output_format,
            output_bit_depth=output_bit_depth,
            preview_seconds=preview_seconds,
            platform_target=platform_target,
        )
        mt = "audio/mpeg" if output_format == "mp3" else ("audio/flac" if output_format == "flac" else "audio/wav")
        return FileResponse(result["output_path"], media_type=mt, filename=f"preview.{output_format}")
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))
    finally:
        if os.path.exists(tmp): os.remove(tmp)

# ── Master async ──────────────────────────────────────────────────────────────
@app.post("/master", tags=["Mastering"])
async def master_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_peak: float        = Query(0.95,   ge=0.1,   le=1.0),
    use_lufs_normalize: bool  = Query(False),
    target_lufs: float        = Query(-14.0,  ge=-40.0, le=0.0),
    # Multiband con valores conservadores
    mb_low_crossover: float = Query(250.0, ge=20.0, le=2000.0),
    mb_high_crossover: float = Query(4000.0, ge=500.0, le=20000.0),
    mb_low_threshold: float = Query(0.7, ge=0.0, le=1.0),
    mb_low_ratio: float = Query(2.0, ge=1.0, le=20.0),
    mb_low_attack_ms: float = Query(20.0, ge=0.1, le=200.0),
    mb_low_release_ms: float = Query(150.0, ge=10.0, le=1000.0),
    mb_low_makeup_db: float = Query(0.0, ge=-12.0, le=24.0),
    mb_mid_threshold: float = Query(0.7, ge=0.0, le=1.0),
    mb_mid_ratio: float = Query(2.0, ge=1.0, le=20.0),
    mb_mid_attack_ms: float = Query(20.0, ge=0.1, le=200.0),
    mb_mid_release_ms: float = Query(150.0, ge=10.0, le=1000.0),
    mb_mid_makeup_db: float = Query(0.0, ge=-12.0, le=24.0),
    mb_high_threshold: float = Query(0.7, ge=0.0, le=1.0),
    mb_high_ratio: float = Query(2.0, ge=1.0, le=20.0),
    mb_high_attack_ms: float = Query(20.0, ge=0.1, le=200.0),
    mb_high_release_ms: float = Query(150.0, ge=10.0, le=1000.0),
    mb_high_makeup_db: float = Query(0.0, ge=-12.0, le=24.0),
    mb_bypass: bool = Query(True),
    input_gain_db: float      = Query(0.0,    ge=-24.0, le=24.0),
    comp_threshold: float      = Query(0.5,   ge=0.0,   le=1.0),
    comp_ratio: float          = Query(4.0,   ge=1.0,   le=20.0),
    comp_attack_ms: float      = Query(10.0,  ge=0.1,   le=200.0),
    comp_release_ms: float     = Query(100.0, ge=10.0,  le=1000.0),
    comp_makeup_db: float      = Query(0.0,   ge=-12.0, le=24.0),
    comp_stereo_link: bool     = Query(True, description="Linkea L/R en el compresor para preservar la imagen estéreo"),
    oversample_mode: str       = Query("quality", pattern="^(off|draft|fast|quality|ultra)$"),
    # EQ
    hp_cutoff: float          = Query(80.0,   ge=20.0,  le=500.0),
    high_shelf_gain_db: float = Query(2.0,    ge=-12.0, le=12.0),
    high_shelf_freq_hz: float  = Query(8000.0, ge=1000.0, le=20000.0),
    # Multiband Stereo Width
    mb_stereo_bypass: bool          = Query(True),
    mb_stereo_low_width: float      = Query(0.9,   ge=0.0, le=3.0),
    mb_stereo_mid_width: float      = Query(1.2,   ge=0.0, le=3.0),
    mb_stereo_high_width: float     = Query(1.5,   ge=0.0, le=3.0),
    mb_stereo_low_crossover: float  = Query(150.0, ge=20.0, le=2000.0),
    mb_stereo_high_crossover: float = Query(4000.0,ge=200.0, le=20000.0),
    eq1_freq: float = Query(100.0), eq1_gain: float = Query(0.0), eq1_q: float = Query(1.0),
    eq2_freq: float = Query(500.0), eq2_gain: float = Query(0.0), eq2_q: float = Query(1.0),
    eq3_freq: float = Query(2000.0), eq3_gain: float = Query(0.0), eq3_q: float = Query(1.0),
    eq4_freq: float = Query(8000.0), eq4_gain: float = Query(0.0), eq4_q: float = Query(1.0),
    transient_attack: float   = Query(0.0,   ge=-1.0,  le=1.0),
    transient_sustain: float  = Query(0.0,   ge=-1.0,  le=1.0),
    saturation_drive: float   = Query(0.0,   ge=0.0,   le=1.0),
    saturation_mode: str      = Query("tape", pattern="^(tape|tube)$"),
    saturation_mix: float     = Query(1.0,   ge=0.0,   le=1.0),
    mid_gain_db: float        = Query(0.0,   ge=-12.0, le=12.0),
    side_gain_db: float       = Query(0.0,   ge=-18.0, le=18.0),
    stereo_width_amount: float = Query(1.2,  ge=0.0,   le=3.0),
    use_stereo_enhancer: bool  = Query(False),
    enhancer_bass_mono_freq: float = Query(120.0),
    haas_delay_ms: float      = Query(0.0,   ge=0.0,   le=30.0),
    reverb_size: float        = Query(0.3,   ge=0.05,  le=2.0),
    reverb_wet: float         = Query(0.0,   ge=0.0,   le=1.0),
    glue_bypass: bool         = Query(True),
    glue_threshold_db: float  = Query(-4.0,  ge=-24.0, le=0.0),
    glue_ratio: float         = Query(2.0,   ge=1.0,   le=10.0),
    glue_attack_ms: float     = Query(30.0,  ge=0.1,   le=200.0),
    glue_release_ms: float    = Query(120.0, ge=10.0,  le=1000.0),
    glue_makeup_db: float     = Query(0.0,   ge=-12.0, le=12.0),
    limiter_ceiling: float    = Query(0.95,  ge=0.5,   le=1.0),
    limiter_release_ms: float = Query(50.0,  ge=1.0,   le=500.0),
    # EQ de fase lineal (FIR) / Dynamic EQ / Low-End Mono Maker dedicado
    eq_mode: str              = Query("iir", pattern="^(iir|linear_phase)$"),
    linear_phase_taps: int    = Query(2049, ge=257, le=8193),
    low_end_mono_freq: float  = Query(120.0, ge=40.0, le=300.0),
    low_end_mono_amount: float = Query(0.0, ge=0.0, le=1.0),
    dyneq_bypass: bool        = Query(True),
    dyneq_freq: float         = Query(3000.0, ge=200.0, le=16000.0),
    dyneq_q: float            = Query(2.5,   ge=0.5,  le=12.0),
    dyneq_threshold_db: float = Query(-18.0, ge=-60.0, le=0.0),
    dyneq_ratio: float        = Query(3.0,   ge=1.0,  le=20.0),
    dyneq_attack_ms: float    = Query(3.0,   ge=0.1,  le=100.0),
    dyneq_release_ms: float   = Query(80.0,  ge=5.0,  le=1000.0),
    dyneq_max_reduction_db: float = Query(12.0, ge=0.0, le=30.0),
    nr_bypass: bool           = Query(True,  description="Desactivar para aplicar reducción de ruido antes de la cadena."),
    nr_strength: float        = Query(0.5,   ge=0.0, le=1.0, description="Intensidad de la reducción de ruido (0=nada, 1=máximo)."),
    nr_noise_sample_sec: float = Query(0.5,  ge=0.1, le=5.0, description="Segundos iniciales usados para estimar el perfil de ruido."),
    output_format: str        = Query("wav",  pattern="^(wav|flac|mp3)$"),
    output_bit_depth: int     = Query(24, description="Bit depth de salida (WAV/FLAC): 16, 24 o 32 (float). Se aplica dither TPDF si baja de 32."),
    platform_target: str      = Query(None,   pattern="^(spotify|youtube|apple_music|tidal|club|cd)$"),
):
    validate_audio_file(file.filename)
    data = await read_and_validate(file)
    job_id = uuid.uuid4().hex
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    with open(input_path, "wb") as f: f.write(data)

    params = dict(
        target_peak=target_peak,
        use_lufs_normalize=use_lufs_normalize,
        target_lufs=target_lufs,
        input_gain_db=input_gain_db,
        oversample_mode=oversample_mode,
        comp_stereo_link=comp_stereo_link,
        comp_threshold=comp_threshold,
        comp_ratio=comp_ratio,
        comp_attack_ms=comp_attack_ms,
        comp_release_ms=comp_release_ms,
        comp_makeup_db=comp_makeup_db,
        mb_low_crossover=mb_low_crossover,
        mb_high_crossover=mb_high_crossover,
        mb_low_threshold=mb_low_threshold,
        mb_low_ratio=mb_low_ratio,
        mb_low_attack_ms=mb_low_attack_ms,
        mb_low_release_ms=mb_low_release_ms,
        mb_low_makeup_db=mb_low_makeup_db,
        mb_mid_threshold=mb_mid_threshold,
        mb_mid_ratio=mb_mid_ratio,
        mb_mid_attack_ms=mb_mid_attack_ms,
        mb_mid_release_ms=mb_mid_release_ms,
        mb_mid_makeup_db=mb_mid_makeup_db,
        mb_high_threshold=mb_high_threshold,
        mb_high_ratio=mb_high_ratio,
        mb_high_attack_ms=mb_high_attack_ms,
        mb_high_release_ms=mb_high_release_ms,
        mb_high_makeup_db=mb_high_makeup_db,
        mb_bypass=mb_bypass,
        hp_cutoff=hp_cutoff,
        high_shelf_gain_db=high_shelf_gain_db,
        high_shelf_freq_hz=high_shelf_freq_hz,
        mb_stereo_bypass=mb_stereo_bypass,
        mb_stereo_low_width=mb_stereo_low_width,
        mb_stereo_mid_width=mb_stereo_mid_width,
        mb_stereo_high_width=mb_stereo_high_width,
        mb_stereo_low_crossover=mb_stereo_low_crossover,
        mb_stereo_high_crossover=mb_stereo_high_crossover,
        eq1_freq=eq1_freq, eq1_gain=eq1_gain, eq1_q=eq1_q,
        eq2_freq=eq2_freq, eq2_gain=eq2_gain, eq2_q=eq2_q,
        eq3_freq=eq3_freq, eq3_gain=eq3_gain, eq3_q=eq3_q,
        eq4_freq=eq4_freq, eq4_gain=eq4_gain, eq4_q=eq4_q,
        transient_attack=transient_attack,
        transient_sustain=transient_sustain,
        saturation_drive=saturation_drive,
        saturation_mode=saturation_mode,
        saturation_mix=saturation_mix,
        mid_gain_db=mid_gain_db,
        side_gain_db=side_gain_db,
        stereo_width_amount=stereo_width_amount,
        use_stereo_enhancer=use_stereo_enhancer,
        enhancer_bass_mono_freq=enhancer_bass_mono_freq,
        haas_delay_ms=haas_delay_ms,
        reverb_size=reverb_size,
        reverb_wet=reverb_wet,
        glue_bypass=glue_bypass,
        glue_threshold_db=glue_threshold_db,
        glue_ratio=glue_ratio,
        glue_attack_ms=glue_attack_ms,
        glue_release_ms=glue_release_ms,
        glue_makeup_db=glue_makeup_db,
        limiter_ceiling=limiter_ceiling,
        limiter_release_ms=limiter_release_ms,
        eq_mode=eq_mode,
        linear_phase_taps=linear_phase_taps,
        low_end_mono_freq=low_end_mono_freq,
        low_end_mono_amount=low_end_mono_amount,
        dyneq_bypass=dyneq_bypass,
        dyneq_freq=dyneq_freq,
        dyneq_q=dyneq_q,
        dyneq_threshold_db=dyneq_threshold_db,
        dyneq_ratio=dyneq_ratio,
        dyneq_attack_ms=dyneq_attack_ms,
        dyneq_release_ms=dyneq_release_ms,
        dyneq_max_reduction_db=dyneq_max_reduction_db,
        nr_bypass=nr_bypass,
        nr_strength=nr_strength,
        nr_noise_sample_sec=nr_noise_sample_sec,
        output_format=output_format,
        output_bit_depth=output_bit_depth,
        platform_target=platform_target,
    )
    duration = _get_input_duration(input_path)
    job_params = dict(params)
    if duration is not None:
        job_params["_input_duration_sec"] = duration
    jobs[job_id] = {"status": "queued", "filename": file.filename, "created_at": time.time(), "params": job_params, "progress": 0, "stage": "En cola"}
    background_tasks.add_task(run_mastering_job, job_id, input_path, params)
    return {"job_id": job_id, "status": "queued", "poll_url": f"/job/{job_id}"}

# ── Master sync ──────────────────────────────────────────────────────────────
@app.post("/master/sync", tags=["Mastering"])
async def master_sync(
    file: UploadFile = File(...),
    target_peak: float        = Query(0.95,   ge=0.1,   le=1.0),
    use_lufs_normalize: bool  = Query(False),
    target_lufs: float        = Query(-14.0,  ge=-40.0, le=0.0),
    # Multiband con valores conservadores
    mb_low_crossover: float = Query(250.0, ge=20.0, le=2000.0),
    mb_high_crossover: float = Query(4000.0, ge=500.0, le=20000.0),
    mb_low_threshold: float = Query(0.7, ge=0.0, le=1.0),
    mb_low_ratio: float = Query(2.0, ge=1.0, le=20.0),
    mb_low_attack_ms: float = Query(20.0, ge=0.1, le=200.0),
    mb_low_release_ms: float = Query(150.0, ge=10.0, le=1000.0),
    mb_low_makeup_db: float = Query(0.0, ge=-12.0, le=24.0),
    mb_mid_threshold: float = Query(0.7, ge=0.0, le=1.0),
    mb_mid_ratio: float = Query(2.0, ge=1.0, le=20.0),
    mb_mid_attack_ms: float = Query(20.0, ge=0.1, le=200.0),
    mb_mid_release_ms: float = Query(150.0, ge=10.0, le=1000.0),
    mb_mid_makeup_db: float = Query(0.0, ge=-12.0, le=24.0),
    mb_high_threshold: float = Query(0.7, ge=0.0, le=1.0),
    mb_high_ratio: float = Query(2.0, ge=1.0, le=20.0),
    mb_high_attack_ms: float = Query(20.0, ge=0.1, le=200.0),
    mb_high_release_ms: float = Query(150.0, ge=10.0, le=1000.0),
    mb_high_makeup_db: float = Query(0.0, ge=-12.0, le=24.0),
    mb_bypass: bool = Query(True),
    input_gain_db: float      = Query(0.0,    ge=-24.0, le=24.0),
    comp_threshold: float      = Query(0.5,   ge=0.0,   le=1.0),
    comp_ratio: float          = Query(4.0,   ge=1.0,   le=20.0),
    comp_attack_ms: float      = Query(10.0,  ge=0.1,   le=200.0),
    comp_release_ms: float     = Query(100.0, ge=10.0,  le=1000.0),
    comp_makeup_db: float      = Query(0.0,   ge=-12.0, le=24.0),
    comp_stereo_link: bool     = Query(True, description="Linkea L/R en el compresor para preservar la imagen estéreo"),
    oversample_mode: str       = Query("quality", pattern="^(off|draft|fast|quality|ultra)$"),
    # EQ
    hp_cutoff: float          = Query(80.0,   ge=20.0,  le=500.0),
    high_shelf_gain_db: float = Query(2.0,    ge=-12.0, le=12.0),
    high_shelf_freq_hz: float  = Query(8000.0, ge=1000.0, le=20000.0),
    # Multiband Stereo Width
    mb_stereo_bypass: bool          = Query(True),
    mb_stereo_low_width: float      = Query(0.9,   ge=0.0, le=3.0),
    mb_stereo_mid_width: float      = Query(1.2,   ge=0.0, le=3.0),
    mb_stereo_high_width: float     = Query(1.5,   ge=0.0, le=3.0),
    mb_stereo_low_crossover: float  = Query(150.0, ge=20.0, le=2000.0),
    mb_stereo_high_crossover: float = Query(4000.0,ge=200.0, le=20000.0),
    eq1_freq: float = Query(100.0), eq1_gain: float = Query(0.0), eq1_q: float = Query(1.0),
    eq2_freq: float = Query(500.0), eq2_gain: float = Query(0.0), eq2_q: float = Query(1.0),
    eq3_freq: float = Query(2000.0), eq3_gain: float = Query(0.0), eq3_q: float = Query(1.0),
    eq4_freq: float = Query(8000.0), eq4_gain: float = Query(0.0), eq4_q: float = Query(1.0),
    transient_attack: float   = Query(0.0,   ge=-1.0,  le=1.0),
    transient_sustain: float  = Query(0.0,   ge=-1.0,  le=1.0),
    saturation_drive: float   = Query(0.0,   ge=0.0,   le=1.0),
    saturation_mode: str      = Query("tape", pattern="^(tape|tube)$"),
    saturation_mix: float     = Query(1.0,   ge=0.0,   le=1.0),
    mid_gain_db: float        = Query(0.0,   ge=-12.0, le=12.0),
    side_gain_db: float       = Query(0.0,   ge=-18.0, le=18.0),
    stereo_width_amount: float = Query(1.2,  ge=0.0,   le=3.0),
    use_stereo_enhancer: bool  = Query(False),
    enhancer_bass_mono_freq: float = Query(120.0),
    haas_delay_ms: float      = Query(0.0,   ge=0.0,   le=30.0),
    reverb_size: float        = Query(0.3,   ge=0.05,  le=2.0),
    reverb_wet: float         = Query(0.0,   ge=0.0,   le=1.0),
    glue_bypass: bool         = Query(True),
    glue_threshold_db: float  = Query(-4.0,  ge=-24.0, le=0.0),
    glue_ratio: float         = Query(2.0,   ge=1.0,   le=10.0),
    glue_attack_ms: float     = Query(30.0,  ge=0.1,   le=200.0),
    glue_release_ms: float    = Query(120.0, ge=10.0,  le=1000.0),
    glue_makeup_db: float     = Query(0.0,   ge=-12.0, le=12.0),
    limiter_ceiling: float    = Query(0.95,  ge=0.5,   le=1.0),
    limiter_release_ms: float = Query(50.0,  ge=1.0,   le=500.0),
    # EQ de fase lineal (FIR) / Dynamic EQ / Low-End Mono Maker dedicado
    eq_mode: str              = Query("iir", pattern="^(iir|linear_phase)$"),
    linear_phase_taps: int    = Query(2049, ge=257, le=8193),
    low_end_mono_freq: float  = Query(120.0, ge=40.0, le=300.0),
    low_end_mono_amount: float = Query(0.0, ge=0.0, le=1.0),
    dyneq_bypass: bool        = Query(True),
    dyneq_freq: float         = Query(3000.0, ge=200.0, le=16000.0),
    dyneq_q: float            = Query(2.5,   ge=0.5,  le=12.0),
    dyneq_threshold_db: float = Query(-18.0, ge=-60.0, le=0.0),
    dyneq_ratio: float        = Query(3.0,   ge=1.0,  le=20.0),
    dyneq_attack_ms: float    = Query(3.0,   ge=0.1,  le=100.0),
    dyneq_release_ms: float   = Query(80.0,  ge=5.0,  le=1000.0),
    dyneq_max_reduction_db: float = Query(12.0, ge=0.0, le=30.0),
    nr_bypass: bool           = Query(True,  description="Desactivar para aplicar reducción de ruido antes de la cadena."),
    nr_strength: float        = Query(0.5,   ge=0.0, le=1.0, description="Intensidad de la reducción de ruido (0=nada, 1=máximo)."),
    nr_noise_sample_sec: float = Query(0.5,  ge=0.1, le=5.0, description="Segundos iniciales usados para estimar el perfil de ruido."),
    output_format: str        = Query("wav",  pattern="^(wav|flac|mp3)$"),
    output_bit_depth: int     = Query(24, description="Bit depth de salida (WAV/FLAC): 16, 24 o 32 (float). Se aplica dither TPDF si baja de 32."),
    platform_target: str      = Query(None,   pattern="^(spotify|youtube|apple_music|tidal|club|cd)$"),
):
    validate_audio_file(file.filename)
    data = await read_and_validate(file)
    tmp = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{file.filename}")
    try:
        cleanup_old()
        with open(tmp, "wb") as f: f.write(data)
        # BUGFIX: mismo problema de bloqueo del event loop que en /preview.
        result = await run_in_threadpool(
            process_audio,
            tmp,
            target_peak=target_peak,
            use_lufs_normalize=use_lufs_normalize,
            target_lufs=target_lufs,
            input_gain_db=input_gain_db,
            oversample_mode=oversample_mode,
            comp_stereo_link=comp_stereo_link,
            comp_threshold=comp_threshold,
            comp_ratio=comp_ratio,
            comp_attack_ms=comp_attack_ms,
            comp_release_ms=comp_release_ms,
            comp_makeup_db=comp_makeup_db,
            mb_low_crossover=mb_low_crossover,
            mb_high_crossover=mb_high_crossover,
            mb_low_threshold=mb_low_threshold,
            mb_low_ratio=mb_low_ratio,
            mb_low_attack_ms=mb_low_attack_ms,
            mb_low_release_ms=mb_low_release_ms,
            mb_low_makeup_db=mb_low_makeup_db,
            mb_mid_threshold=mb_mid_threshold,
            mb_mid_ratio=mb_mid_ratio,
            mb_mid_attack_ms=mb_mid_attack_ms,
            mb_mid_release_ms=mb_mid_release_ms,
            mb_mid_makeup_db=mb_mid_makeup_db,
            mb_high_threshold=mb_high_threshold,
            mb_high_ratio=mb_high_ratio,
            mb_high_attack_ms=mb_high_attack_ms,
            mb_high_release_ms=mb_high_release_ms,
            mb_high_makeup_db=mb_high_makeup_db,
            mb_bypass=mb_bypass,
            hp_cutoff=hp_cutoff,
            high_shelf_gain_db=high_shelf_gain_db,
            high_shelf_freq_hz=high_shelf_freq_hz,
            mb_stereo_bypass=mb_stereo_bypass,
            mb_stereo_low_width=mb_stereo_low_width,
            mb_stereo_mid_width=mb_stereo_mid_width,
            mb_stereo_high_width=mb_stereo_high_width,
            mb_stereo_low_crossover=mb_stereo_low_crossover,
            mb_stereo_high_crossover=mb_stereo_high_crossover,
            eq1_freq=eq1_freq, eq1_gain=eq1_gain, eq1_q=eq1_q,
            eq2_freq=eq2_freq, eq2_gain=eq2_gain, eq2_q=eq2_q,
            eq3_freq=eq3_freq, eq3_gain=eq3_gain, eq3_q=eq3_q,
            eq4_freq=eq4_freq, eq4_gain=eq4_gain, eq4_q=eq4_q,
            transient_attack=transient_attack,
            transient_sustain=transient_sustain,
            saturation_drive=saturation_drive,
            saturation_mode=saturation_mode,
            saturation_mix=saturation_mix,
            mid_gain_db=mid_gain_db,
            side_gain_db=side_gain_db,
            stereo_width_amount=stereo_width_amount,
            use_stereo_enhancer=use_stereo_enhancer,
            enhancer_bass_mono_freq=enhancer_bass_mono_freq,
            haas_delay_ms=haas_delay_ms,
            reverb_size=reverb_size,
            reverb_wet=reverb_wet,
            glue_bypass=glue_bypass,
            glue_threshold_db=glue_threshold_db,
            glue_ratio=glue_ratio,
            glue_attack_ms=glue_attack_ms,
            glue_release_ms=glue_release_ms,
            glue_makeup_db=glue_makeup_db,
            limiter_ceiling=limiter_ceiling,
            limiter_release_ms=limiter_release_ms,
            eq_mode=eq_mode,
            linear_phase_taps=linear_phase_taps,
            low_end_mono_freq=low_end_mono_freq,
            low_end_mono_amount=low_end_mono_amount,
            dyneq_bypass=dyneq_bypass,
            dyneq_freq=dyneq_freq,
            dyneq_q=dyneq_q,
            dyneq_threshold_db=dyneq_threshold_db,
            dyneq_ratio=dyneq_ratio,
            dyneq_attack_ms=dyneq_attack_ms,
            dyneq_release_ms=dyneq_release_ms,
            dyneq_max_reduction_db=dyneq_max_reduction_db,
            nr_bypass=nr_bypass,
            nr_strength=nr_strength,
            nr_noise_sample_sec=nr_noise_sample_sec,
            output_format=output_format,
            output_bit_depth=output_bit_depth,
            platform_target=platform_target,
        )
        mt = "audio/mpeg" if output_format == "mp3" else ("audio/flac" if output_format == "flac" else "audio/wav")
        return FileResponse(result["output_path"], media_type=mt, filename=f"mastered.{output_format}")
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))
    finally:
        if os.path.exists(tmp): os.remove(tmp)

# ── Master por referencia (reference-track matching) ───────────────────────────
def _read_reference_params(
    eq_bands: int, eq_max_boost_db: float, eq_max_cut_db: float, eq_q: float,
    eq_match_blend: float, oversample_mode: str,
    match_loudness: bool, match_dynamics: bool, match_stereo_width: bool,
    hp_cutoff: float, limiter_release_ms: float, output_format: str,
    output_bit_depth: int = 24,
    dynamics_margin_db: float = 1.0, stereo_blend: float = 0.85,
) -> dict:
    return dict(
        eq_bands=eq_bands, eq_max_boost_db=eq_max_boost_db, eq_max_cut_db=eq_max_cut_db,
        eq_q=eq_q, eq_match_blend=eq_match_blend, oversample_mode=oversample_mode,
        match_loudness=match_loudness, match_dynamics=match_dynamics,
        match_stereo_width=match_stereo_width, hp_cutoff=hp_cutoff,
        limiter_release_ms=limiter_release_ms, output_format=output_format,
        output_bit_depth=output_bit_depth,
        dynamics_margin_db=dynamics_margin_db, stereo_blend=stereo_blend,
    )

@app.post("/master/reference", tags=["Mastering"])
async def master_with_reference(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Track propio a masterizar"),
    reference_file: UploadFile = File(..., description="Track de referencia (sonido objetivo)"),
    eq_bands: int              = Query(28,   ge=4,    le=40),
    eq_max_boost_db: float     = Query(6.0,  ge=0.0,  le=18.0),
    eq_max_cut_db: float       = Query(-9.0, ge=-24.0, le=0.0),
    eq_q: float                = Query(1.3,  ge=0.3,  le=6.0),
    eq_match_blend: float      = Query(0.75, ge=0.0,  le=1.0,
                                       description="Cantidad de EQ match a aplicar (0=no toca, 1=matching completo)"),
    oversample_mode: str       = Query("quality", pattern="^(off|draft|fast|quality|ultra)$"),
    match_loudness: bool       = Query(True),
    match_dynamics: bool       = Query(True),
    match_stereo_width: bool   = Query(True),
    hp_cutoff: float           = Query(30.0, ge=20.0, le=200.0),
    limiter_release_ms: float  = Query(60.0, ge=1.0,  le=500.0),
    output_format: str         = Query("wav", pattern="^(wav|flac|mp3)$"),
    output_bit_depth: int      = Query(24, description="Bit depth de salida (WAV/FLAC): 16, 24 o 32 (float). Se aplica dither TPDF si baja de 32."),
    dynamics_margin_db: float  = Query(1.0,  ge=0.0,  le=6.0,
                                       description="Margen (dB) de crest factor por banda antes de comprimir para acercar la dinámica a la referencia"),
    stereo_blend: float        = Query(0.85, ge=0.0,  le=1.0,
                                       description="Qué tan agresivamente se acerca el ancho estéreo por banda al de la referencia (0=no toca, 1=matching total)"),
):
    validate_audio_file(file.filename)
    validate_audio_file(reference_file.filename)
    data     = await read_and_validate(file)
    ref_data = await read_and_validate(reference_file)

    job_id = uuid.uuid4().hex
    input_path     = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    reference_path = os.path.join(UPLOAD_DIR, f"{job_id}_ref_{reference_file.filename}")
    with open(input_path, "wb") as f: f.write(data)
    with open(reference_path, "wb") as f: f.write(ref_data)

    params = _read_reference_params(eq_bands, eq_max_boost_db, eq_max_cut_db, eq_q,
                                    eq_match_blend, oversample_mode,
                                    match_loudness, match_dynamics, match_stereo_width,
                                    hp_cutoff, limiter_release_ms, output_format,
                                    output_bit_depth, dynamics_margin_db, stereo_blend)
    duration = _get_input_duration(input_path)
    job_params = dict(params, reference_filename=reference_file.filename)
    if duration is not None:
        job_params["_input_duration_sec"] = duration
    jobs[job_id] = {"status": "queued", "filename": file.filename, "created_at": time.time(), "params": job_params, "progress": 0, "stage": "En cola"}
    background_tasks.add_task(run_reference_job, job_id, input_path, reference_path, params)
    return {"job_id": job_id, "status": "queued", "poll_url": f"/job/{job_id}"}

@app.post("/master/reference/sync", tags=["Mastering"])
async def master_with_reference_sync(
    file: UploadFile = File(..., description="Track propio a masterizar"),
    reference_file: UploadFile = File(..., description="Track de referencia (sonido objetivo)"),
    eq_bands: int              = Query(28,   ge=4,    le=40),
    eq_max_boost_db: float     = Query(6.0,  ge=0.0,  le=18.0),
    eq_max_cut_db: float       = Query(-9.0, ge=-24.0, le=0.0),
    eq_q: float                = Query(1.3,  ge=0.3,  le=6.0),
    eq_match_blend: float      = Query(0.75, ge=0.0,  le=1.0,
                                       description="Cantidad de EQ match a aplicar (0=no toca, 1=matching completo)"),
    oversample_mode: str       = Query("quality", pattern="^(off|draft|fast|quality|ultra)$"),
    match_loudness: bool       = Query(True),
    match_dynamics: bool       = Query(True),
    match_stereo_width: bool   = Query(True),
    hp_cutoff: float           = Query(30.0, ge=20.0, le=200.0),
    limiter_release_ms: float  = Query(60.0, ge=1.0,  le=500.0),
    output_format: str         = Query("wav", pattern="^(wav|flac|mp3)$"),
    output_bit_depth: int      = Query(24, description="Bit depth de salida (WAV/FLAC): 16, 24 o 32 (float). Se aplica dither TPDF si baja de 32."),
    preview_seconds: float     = Query(None, ge=1.0, le=60.0),
    dynamics_margin_db: float  = Query(1.0,  ge=0.0,  le=6.0),
    stereo_blend: float        = Query(0.85, ge=0.0,  le=1.0),
):
    validate_audio_file(file.filename)
    validate_audio_file(reference_file.filename)
    data     = await read_and_validate(file)
    ref_data = await read_and_validate(reference_file)

    tmp     = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{file.filename}")
    tmp_ref = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_ref_{reference_file.filename}")
    try:
        cleanup_old()
        with open(tmp, "wb") as f: f.write(data)
        with open(tmp_ref, "wb") as f: f.write(ref_data)
        result = await run_in_threadpool(
            process_audio_with_reference, tmp, tmp_ref,
            eq_bands=eq_bands, eq_max_boost_db=eq_max_boost_db, eq_max_cut_db=eq_max_cut_db,
            eq_q=eq_q, eq_match_blend=eq_match_blend, oversample_mode=oversample_mode,
            match_loudness=match_loudness, match_dynamics=match_dynamics,
            match_stereo_width=match_stereo_width, hp_cutoff=hp_cutoff,
            limiter_release_ms=limiter_release_ms, output_format=output_format,
            output_bit_depth=output_bit_depth,
            preview_seconds=preview_seconds,
            dynamics_margin_db=dynamics_margin_db, stereo_blend=stereo_blend,
        )
        mt = "audio/mpeg" if output_format == "mp3" else ("audio/flac" if output_format == "flac" else "audio/wav")
        headers = {"X-Reference-Match": str(result["reference_match"]["after"]["match_percent"])}
        return FileResponse(result["output_path"], media_type=mt, filename=f"mastered_refmatch.{output_format}", headers=headers)
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))
    finally:
        if os.path.exists(tmp): os.remove(tmp)
        if os.path.exists(tmp_ref): os.remove(tmp_ref)

# ── Jobs ──────────────────────────────────────────────────────────────────────
@app.get("/job/{job_id}", tags=["Jobs"])
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job no encontrado")
    job = jobs[job_id].copy()
    if job.get("type") == "stems":
        # Job de stem separation (#13) — no tiene "result" como los de
        # mastering, ya deja stem_analysis/available_stems seteados en
        # run_stems_job(). Acá solo agregamos las download_url por stem y
        # sacamos stem_paths (son paths de servidor, no deben salir por API).
        if job["status"] == "done":
            job["stem_download_urls"] = {
                name: f"/stems/download/{job_id}/{name}" for name in job.get("available_stems", [])
            }
        job.pop("stem_paths", None)
        return job
    if job["status"] == "done":
        job["download_url"]      = f"/download/{job_id}"
        job["report_url"]        = f"/report/{job_id}"
        job["analysis_before"]   = job["result"]["analysis_before"]
        job["analysis_after"]    = job["result"]["analysis_after"]
        job["mix_advice_before"] = job["result"]["mix_advice_before"]
        job["mix_advice_after"]  = job["result"]["mix_advice_after"]
        job["chain_meters"]      = job["result"].get("chain_meters", {})
        job["output_bit_depth"]  = job["result"].get("output_bit_depth")
        if "reference_match" in job["result"]:
            job["reference_match"]    = job["result"]["reference_match"]
            job["analysis_reference"] = job["result"]["analysis_reference"]
        del job["result"]
    return job

@app.get("/download/{job_id}", tags=["Jobs"])
def download(job_id: str, name: Optional[str] = Query(None, description="Nombre del tema para el archivo descargado")):
    if job_id not in jobs:
        raise HTTPException(404, "Job no encontrado")
    job = jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(400, f"Job no listo: {job['status']}")
    output_path = job["result"]["output_path"]
    if not os.path.exists(output_path):
        raise HTTPException(410, "Archivo expirado. Volvé a masterizar.")
    fmt = job["params"]["output_format"]
    mt = "audio/mpeg" if fmt == "mp3" else ("audio/flac" if fmt == "flac" else "audio/wav")
    track_name = sanitize_track_name(name)
    return FileResponse(output_path, media_type=mt, filename=f"{track_name}.{fmt}")

@app.get("/report/{job_id}", tags=["Jobs"])
def export_report(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job no encontrado")
    job = jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(400, f"Job no listo: {job['status']}")
    report = {
        "job_id": job_id,
        "filename": job["filename"],
        "created_at": job["created_at"],
        "finished_at": job.get("finished_at"),
        "params": job["params"],
        "analysis_before": job["result"]["analysis_before"],
        "analysis_after": job["result"]["analysis_after"],
        "mix_advice_before": job["result"]["mix_advice_before"],
        "mix_advice_after": job["result"]["mix_advice_after"],
        "chain_meters": job["result"].get("chain_meters", {}),
    }
    if "reference_match" in job["result"]:
        report["reference_match"]    = job["result"]["reference_match"]
        report["analysis_reference"] = job["result"]["analysis_reference"]
    return JSONResponse(content=report, headers={
        "Content-Disposition": f'attachment; filename="mastering_report_{job_id[:8]}.json"'
    })

@app.get("/jobs", tags=["Jobs"])
def list_jobs():
    return [{"job_id": k, "status": v["status"], "filename": v["filename"], "created_at": v["created_at"]}
            for k, v in jobs.items()]