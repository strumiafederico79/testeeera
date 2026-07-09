"""
INTEGRACIÓN — pegar en app.py.

No tengo tu app.py real en este chat (solo subiste streaming_engine.py +
index.html), así que este snippet asume el patrón que ya usás en el resto
de la app (JOBS dict en memoria + BackgroundTasks + /job/{id} polling, con
progress/stage — el mismo que ya conectaste para mastering).

Si tus helpers se llaman distinto (ej. create_job / update_job / JOBS),
avisame y te tiro el archivo ya adaptado al 100%.
"""
import io
import numpy as np
import soundfile as sf
from fastapi import UploadFile, File, BackgroundTasks

from stem_separation import separate_stems
from stem_analysis import analyze_stems_full

try:
    from mastering import measure_lufs_integrated as _lufs_fn
except Exception:
    _lufs_fn = None


def _process_stems_job(job_id: str, audio: np.ndarray, sr: int):
    """Worker que corre en background. Adaptar nombres de JOBS/update_job."""
    try:
        def progress_cb(pct, stage):
            JOBS[job_id]["progress"] = pct
            JOBS[job_id]["stage"] = stage
            JOBS[job_id]["status"] = "processing"

        JOBS[job_id] = {**JOBS.get(job_id, {}), "status": "processing", "progress": 0, "stage": "Iniciando…"}

        stems = separate_stems(audio, sr, progress_cb=progress_cb)

        JOBS[job_id]["stage"] = "Analizando stems…"
        JOBS[job_id]["progress"] = 96
        result = analyze_stems_full(stems, sr, lufs_fn=_lufs_fn)

        # Guardar los stems en memoria/disco para poder descargarlos después.
        # Ejemplo simple: WAV en memoria, cacheado por job_id (ajustar a tu
        # esquema real de almacenamiento/descarga, ej. carpeta /tmp/jobs/{id}/).
        stem_bytes = {}
        for name, stem_audio in stems.items():
            buf = io.BytesIO()
            sf.write(buf, stem_audio.T if stem_audio.ndim == 2 else stem_audio, sr, format="WAV", subtype="PCM_24")
            stem_bytes[name] = buf.getvalue()
        STEM_FILES[job_id] = stem_bytes  # dict global análogo a JOBS

        JOBS[job_id].update({
            "status": "done",
            "progress": 100,
            "stage": "Listo",
            "stem_analysis": result,
            "available_stems": list(stems.keys()),
        })
    except Exception as e:
        JOBS[job_id] = {**JOBS.get(job_id, {}), "status": "error", "error": str(e)}


@app.post("/stems/separate")
async def stems_separate(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    import uuid
    audio, sr = sf.read(io.BytesIO(await file.read()), always_2d=True)
    audio = audio.T.astype(np.float32)  # (channels, samples)

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "queued", "progress": 0, "stage": "En cola…"}
    background_tasks.add_task(_process_stems_job, job_id, audio, sr)
    return {"job_id": job_id}


@app.get("/stems/download/{job_id}/{stem_name}")
async def stems_download(job_id: str, stem_name: str):
    from fastapi.responses import Response
    data = STEM_FILES.get(job_id, {}).get(stem_name)
    if data is None:
        from fastapi import HTTPException
        raise HTTPException(404, "Stem no encontrado")
    return Response(content=data, media_type="audio/wav",
                     headers={"Content-Disposition": f'attachment; filename="{stem_name}.wav"'})


# Reusa tu endpoint /job/{id} existente tal cual — ya devuelve progress/stage,
# y ahora además incluirá stem_analysis + available_stems cuando status=="done".
# Si preferís un endpoint separado en vez de reusar /job/{id}, decime y lo separo.

# Agregar cerca de donde definís JOBS:
# STEM_FILES: dict = {}
