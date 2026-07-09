"""
Stem separation (#13) usando Demucs (htdemucs, 4 stems: vocals/drums/bass/other).

Requisitos (agregar a requirements.txt):
    demucs==4.0.1
    torch>=2.0        (si hay GPU: instalar build con CUDA)

Notas de hardware:
- Con GPU (CUDA), un track de ~4 min separa en unos segundos-pocos minutos.
- En CPU puro puede tardar varios minutos por track (htdemucs es pesado).
  Como el VPS no tiene límite de recursos, dejamos device='auto' por defecto
  (usa CUDA si torch.cuda.is_available(), si no cae a CPU).
- El modelo se descarga una única vez (~80-300MB según variante) y se cachea
  en ~/.cache/torch/hub/checkpoints — la primera separación va a ser más lenta.

API pública:
    separate_stems(audio, sr, progress_cb=None, model_name="htdemucs", device=None)
        -> dict[str, np.ndarray]   # {"vocals":..,"drums":..,"bass":..,"other":..}
        Cada array tiene shape (channels, samples) en el sr original de entrada.
"""
import numpy as np

STEM_NAMES = ["drums", "bass", "other", "vocals"]  # orden nativo de htdemucs

_MODEL_CACHE = {}


def _get_device(device):
    import torch
    if device and device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_model(model_name="htdemucs"):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    from demucs.pretrained import get_model
    model = get_model(model_name)
    model.eval()
    _MODEL_CACHE[model_name] = model
    return model


def _resample(audio_2d: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio_2d
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(int(sr_in), int(sr_out))
    up, down = sr_out // g, sr_in // g
    return resample_poly(audio_2d, up, down, axis=-1).astype(np.float32)


def separate_stems(audio: np.ndarray, sr: int, progress_cb=None,
                    model_name: str = "htdemucs", device: str = None) -> dict:
    """
    audio: np.ndarray mono (samples,) o estéreo (2, samples) o (samples, 2).
    progress_cb: callable(pct: float, stage: str) -> None, pct en [0,100].
    Devuelve dict {stem_name: np.ndarray (channels, samples)} en sr original.
    """
    import torch
    from demucs.apply import apply_model

    def _report(pct, stage):
        if progress_cb:
            try:
                progress_cb(min(99.0, max(0.0, pct)), stage)
            except Exception:
                pass

    _report(1, "Cargando modelo Demucs…")
    model = _get_model(model_name)
    dev = _get_device(device)
    model.to(dev)
    model_sr = model.samplerate  # típicamente 44100

    # Normalizar shape a (channels, samples), forzar estéreo (demucs espera 2ch)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio_2d = np.stack([audio, audio], axis=0)
    elif audio.ndim == 2:
        audio_2d = audio if audio.shape[0] <= 2 else audio.T
        if audio_2d.shape[0] == 1:
            audio_2d = np.concatenate([audio_2d, audio_2d], axis=0)
    else:
        raise ValueError("audio debe ser mono o estéreo")

    _report(3, "Remuestreando…")
    audio_model_sr = _resample(audio_2d, sr, model_sr)

    wav = torch.from_numpy(audio_model_sr)
    ref_mean = wav.mean()
    ref_std = wav.std() + 1e-8
    wav_norm = (wav - ref_mean) / ref_std

    _report(5, "Separando stems (esto puede tardar varios minutos)…")
    # NOTA: demucs==4.0.1 (última versión publicada en PyPI) NO soporta un
    # parámetro `callback` en apply_model — esa firma es de una versión más
    # nueva que todavía no está en PyPI. Por eso acá no hay progreso granular
    # durante la separación en sí: pasamos de 5% a 90% de una sola vez. Si en
    # el futuro actualizan demucs a una versión con callback, se puede volver
    # a conectar progress_cb en cada segmento.
    with torch.no_grad():
        out = apply_model(
            model, wav_norm[None], device=dev, progress=False,
            split=True, overlap=0.25, shifts=1,
        )[0]
    _report(90, "Separación completa, reconstruyendo stems…")

    out = out * ref_std + ref_mean
    out_np = out.cpu().numpy().astype(np.float32)  # (n_stems, channels, samples)

    # Evita deadlocks OpenMP entre los threads internos de torch (usados por
    # Demucs) y los de scipy (usados después en stem_analysis.py, en el mismo
    # proceso). Es un problema conocido cuando ambas libs corren en CPU y
    # comparten runtime OpenMP — bajar torch a 1 thread ni bien terminamos de
    # usarlo evita que se cuelgue el paso de análisis que viene justo después.
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    _report(97, "Remuestreando stems al sample rate original…")
    stems = {}
    for i, name in enumerate(model.sources):
        stem_audio = _resample(out_np[i], model_sr, sr)
        stems[name] = stem_audio

    # Reordenar a orden canónico si el modelo trae otro orden de sources
    stems = {name: stems[name] for name in STEM_NAMES if name in stems}

    _report(100, "Separación completa")
    return stems
