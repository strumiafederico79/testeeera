"""
Motor de mastering en tiempo real / streaming.
Procesa el audio en bloques (chunks) entregando PCM16 y métricas completas
(LUFS, peak, RMS, correlación estéreo, GR multibanda, GR banda ancha y
glue) para visualización en vivo sin bloquear la interfaz.
"""
import numpy as np
from mastering import apply_mastering_chain, measure_lufs_integrated, stereo_correlation

CHUNK_SECONDS_DEFAULT = 2.0
DEFAULT_OVERLAP_SECONDS = 0.02


def iter_mastering_chunks(audio: np.ndarray, sr: int,
                          chunk_seconds: float = CHUNK_SECONDS_DEFAULT,
                          overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
                          **chain_params):
    """
    Yields (processed_block, metrics_dict) for each chunk.

    Para reducir artefactos entre bloques, el motor procesa cada chunk con un
    pequeño contexto del chunk anterior (solapamiento temporal). Esto mejora la
    continuidad del procesamiento sin cambiar la API pública.
    """
    if audio is None:
        return

    if audio.ndim == 1:
        audio_2d = audio[np.newaxis, :]
        input_is_mono = True
    elif audio.ndim == 2:
        audio_2d = audio
        input_is_mono = False
    else:
        raise ValueError("audio debe ser mono o estéreo (1D/2D)")

    total_samples = int(audio_2d.shape[-1])
    chunk_samples = max(1, int(chunk_seconds * sr))
    overlap_samples = max(0, int(overlap_seconds * sr))
    overlap_samples = min(overlap_samples, max(1, chunk_samples // 2))
    n_chunks = int(np.ceil(total_samples / chunk_samples))

    context = np.zeros((audio_2d.shape[0], overlap_samples), dtype=np.float32)

    for i in range(n_chunks):
        start = i * chunk_samples
        end = min(start + chunk_samples, total_samples)
        block = audio_2d[:, start:end]
        if block.shape[-1] == 0:
            continue

        block = np.asarray(block, dtype=np.float32)
        if overlap_samples > 0 and context.shape[-1] > 0:
            combined = np.concatenate([context, block], axis=1)
        else:
            combined = block

        processed, chain_meters = apply_mastering_chain(combined, sr, **chain_params)
        processed = np.asarray(processed, dtype=np.float32)

        if overlap_samples > 0 and processed.shape[-1] > overlap_samples:
            out_block = processed[:, overlap_samples:overlap_samples + block.shape[-1]]
        else:
            out_block = processed[:, :block.shape[-1]]

        if out_block.shape[-1] < block.shape[-1]:
            pad = np.zeros((out_block.shape[0], block.shape[-1] - out_block.shape[-1]), dtype=np.float32)
            out_block = np.concatenate([out_block, pad], axis=1)

        if input_is_mono:
            out_block = out_block[0]

        mono = out_block.mean(axis=0) if out_block.ndim == 2 else out_block
        mono = np.asarray(mono, dtype=np.float32)
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        rms = float(np.sqrt(np.mean(mono ** 2))) if mono.size else 0.0
        peak_db = float(20.0 * np.log10(peak + 1e-9)) if peak > 0.0 else -120.0
        rms_db = float(20.0 * np.log10(rms + 1e-9)) if rms > 0.0 else -120.0

        try:
            lufs_chunk = measure_lufs_integrated(out_block, sr)
            if not np.isfinite(lufs_chunk):
                raise ValueError("LUFS no finito")
        except Exception:
            lufs_chunk = rms_db - 0.691

        try:
            corr = stereo_correlation(out_block)
        except Exception:
            corr = 0.0

        # FFT compacta para visualizador de espectro en tiempo real.
        # 32 bandas logarítmicas de 20Hz a 20kHz — suficiente para un
        # analizador visual sin saturar el WebSocket.
        try:
            N_FFT = 2048
            mono_fft = (out_block.mean(axis=0) if out_block.ndim == 2 else out_block).astype(np.float32)
            if len(mono_fft) >= N_FFT:
                frame = mono_fft[-N_FFT:]
            else:
                frame = np.pad(mono_fft, (N_FFT - len(mono_fft), 0))
            window = np.hanning(N_FFT)
            spectrum = np.abs(np.fft.rfft(frame * window))
            freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
            N_BANDS = 32
            lo, hi = 20.0, 20000.0
            edges = np.logspace(np.log10(lo), np.log10(hi), N_BANDS + 1)
            bands_db = []
            for b in range(N_BANDS):
                mask = (freqs >= edges[b]) & (freqs < edges[b + 1])
                val = float(np.mean(spectrum[mask])) if mask.any() else 0.0
                bands_db.append(round(float(20.0 * np.log10(val + 1e-9)), 1))
            spectrum_data = {"bands_db": bands_db, "freq_edges": [round(e, 1) for e in edges.tolist()]}
        except Exception:
            spectrum_data = {}

        metrics = {
            "chunk_index": i,
            "n_chunks": n_chunks,
            "progress_pct": round(((i + 1) / n_chunks) * 100.0, 1),
            "peak_db": round(peak_db, 2),
            "rms_db": round(rms_db, 2),
            "lufs_momentary": round(lufs_chunk, 2),
            "stereo_correlation": round(corr, 3),
            "time_sec": round(start / sr, 2),
            "mb_meters": chain_meters.get("mb", {}),
            "comp_meters": chain_meters.get("comp", {}),
            "glue_meters": chain_meters.get("glue", {}),
            "pre_limiter": chain_meters.get("pre_limiter", {}),
            "post_limiter": chain_meters.get("post_limiter", {}),
            "spectrum": spectrum_data,
        }

        context = block[:, -overlap_samples:] if overlap_samples > 0 else np.zeros((block.shape[0], 0), dtype=np.float32)
        yield out_block, metrics


def master_stream_to_pcm16(audio: np.ndarray, sr: int,
                           chunk_seconds: float = CHUNK_SECONDS_DEFAULT,
                           **chain_params):
    """
    Yields (pcm16_bytes, metrics_dict) for each processed chunk.
    pcm16_bytes: interleaved int16 PCM (little-endian).
    """
    for processed, metrics in iter_mastering_chunks(audio, sr,
                                                    chunk_seconds=chunk_seconds,
                                                    **chain_params):
        block  = processed.T if processed.ndim == 2 else processed
        pcm16  = (np.clip(block, -1.0, 1.0) * 32767.0).astype(np.int16)
        yield pcm16.tobytes(), metrics
