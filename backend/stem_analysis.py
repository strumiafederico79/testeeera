"""
Análisis individual de stems + detección de "masking" (colisiones espectrales
y de envolvente) entre stems, con recomendaciones en texto tipo:
    "El kick está tapando al bajo en ~55-90 Hz"

No depende de mastering.py — usa solo numpy/scipy — para poder importarse
en cualquier contexto (job worker, script standalone, tests).
"""
import numpy as np
from scipy.signal import welch, butter, filtfilt

# Bandas críticas para detección de colisiones genéricas (Hz)
CRITICAL_BANDS = {
    "sub":        (20, 60),
    "low_end":    (60, 150),      # fundamental de kick / bajo
    "low_mid":    (150, 400),     # "boxiness"
    "mid":        (400, 1000),
    "upper_mid":  (1000, 3000),   # presencia / cuerpo vocal
    "presence":   (3000, 6000),   # inteligibilidad, ataque
    "air":        (6000, 16000),
}

STEM_LABELS = {"drums": "🥁 Batería", "bass": "🎸 Bajo", "vocals": "🎤 Voz", "other": "🎹 Otros"}

# Umbral: qué fracción de la energía total de un stem debe caer en una banda
# para considerar que ese stem "vive" ahí.
_COLLISION_ENERGY_THRESHOLD = 0.16
_MIN_OVERLAP_SCORE = 0.10

# Analizar el track entero (que puede ser de varios minutos a sr original)
# con Welch + filtfilt es innecesariamente caro y, combinado con el problema
# de threads de torch/scipy, es lo que estaba colgando el job en el 96%.
# Con 90s del centro del track (donde suele estar el groove ya establecido)
# alcanza de sobra para el perfil espectral y la correlación de envolvente.
_MAX_ANALYSIS_SECONDS = 90.0


def _trim_for_analysis(mono: np.ndarray, sr: int) -> np.ndarray:
    max_samples = int(_MAX_ANALYSIS_SECONDS * sr)
    if mono.size <= max_samples:
        return mono
    start = (mono.size - max_samples) // 2
    return mono[start:start + max_samples]


def _to_mono(stem_audio: np.ndarray) -> np.ndarray:
    a = np.asarray(stem_audio, dtype=np.float32)
    if a.ndim == 2:
        return a.mean(axis=0)
    return a


def _band_energy_profile(mono: np.ndarray, sr: int) -> dict:
    """Devuelve {banda: fracción de energía total} usando Welch PSD."""
    if mono.size < 256:
        return {b: 0.0 for b in CRITICAL_BANDS}
    nperseg = min(8192, mono.size)
    freqs, psd = welch(mono, fs=sr, nperseg=nperseg)
    total = float(np.sum(psd)) + 1e-12
    profile = {}
    for band, (lo, hi) in CRITICAL_BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        profile[band] = float(np.sum(psd[mask]) / total)
    return profile, freqs, psd


def _dominant_freq(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any() or np.sum(psd[mask]) <= 0:
        return 0.0
    idx = np.argmax(psd[mask])
    return float(freqs[mask][idx])


def _low_band_envelope(mono: np.ndarray, sr: int, lo=30, hi=250, hop=512) -> np.ndarray:
    """Envolvente RMS de la banda baja, para medir coincidencia rítmica (kick vs bajo).

    Se downsamplea antes de filtrar: la banda de interés (30-250 Hz) no
    necesita más que ~1kHz de sample rate, y correr filtfilt sobre el sr
    original (44.1/48kHz) es ~10x más caro sin ninguna ganancia real."""
    target_sr = 2000
    if sr > target_sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(sr), target_sr)
        up, down = target_sr // g, sr // g
        mono_ds = resample_poly(mono, up, down)
        ds_sr = target_sr
        hop = max(8, int(round(hop * ds_sr / sr)))
    else:
        mono_ds = mono
        ds_sr = sr

    nyq = ds_sr / 2.0
    b, a = butter(4, [max(lo / nyq, 1e-4), min(hi / nyq, 0.99)], btype="band")
    filtered = filtfilt(b, a, mono_ds)
    n_frames = max(1, len(filtered) // hop)
    env = np.sqrt(np.array([
        np.mean(filtered[i * hop:(i + 1) * hop] ** 2) + 1e-12
        for i in range(n_frames)
    ]))
    return env


def _envelope_correlation(env_a: np.ndarray, env_b: np.ndarray) -> float:
    n = min(len(env_a), len(env_b))
    if n < 4:
        return 0.0
    a, b = env_a[:n], env_b[:n]
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.clip(np.corrcoef(a, b)[0, 1], -1.0, 1.0))


def analyze_stem(name: str, stem_audio: np.ndarray, sr: int, lufs_fn=None) -> dict:
    """Métricas individuales de un stem (peak, rms, lufs opcional, banda dominante, etc.)."""
    mono = _to_mono(stem_audio)
    mono = _trim_for_analysis(mono, sr)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(mono ** 2))) if mono.size else 0.0
    peak_db = 20.0 * np.log10(peak + 1e-9) if peak > 0 else -120.0
    rms_db = 20.0 * np.log10(rms + 1e-9) if rms > 0 else -120.0

    profile, freqs, psd = _band_energy_profile(mono, sr)
    dominant_band = max(profile, key=profile.get) if profile else None

    lufs = None
    if lufs_fn is not None:
        try:
            lufs = float(lufs_fn(stem_audio if stem_audio.ndim == 2 else mono, sr))
        except Exception:
            lufs = None

    return {
        "name": name,
        "label": STEM_LABELS.get(name, name),
        "peak_db": round(peak_db, 2),
        "rms_db": round(rms_db, 2),
        "lufs": round(lufs, 2) if lufs is not None else None,
        "is_silent": rms_db < -55.0,
        "band_energy_profile": {k: round(v, 4) for k, v in profile.items()},
        "dominant_band": dominant_band,
    }


def detect_masking(stems: dict, sr: int) -> list:
    """
    stems: {name: np.ndarray}. Devuelve lista de recomendaciones estructuradas:
    [{"stems":[a,b], "type":..., "band_hz":[lo,hi], "score":.., "message": "..."}]
    """
    mono_stems = {name: _trim_for_analysis(_to_mono(a), sr) for name, a in stems.items() if not _is_silent(a)}
    if len(mono_stems) < 2:
        return []

    profiles, freqs_map, psd_map = {}, {}, {}
    for name, mono in mono_stems.items():
        profile, freqs, psd = _band_energy_profile(mono, sr)
        profiles[name] = profile
        freqs_map[name] = freqs
        psd_map[name] = psd

    recs = []

    # --- Chequeo específico: kick (drums, banda low_end) vs bajo (bass) -------
    if "drums" in mono_stems and "bass" in mono_stems:
        rec = _check_kick_bass(mono_stems, sr, profiles, freqs_map, psd_map)
        if rec:
            recs.append(rec)

    # --- Chequeo genérico: cualquier par de stems que compitan por una banda --
    names = list(mono_stems.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if {a, b} == {"drums", "bass"}:
                continue  # ya cubierto arriba con más detalle
            for band, (lo, hi) in CRITICAL_BANDS.items():
                fa, fb = profiles[a].get(band, 0.0), profiles[b].get(band, 0.0)
                if fa < _COLLISION_ENERGY_THRESHOLD or fb < _COLLISION_ENERGY_THRESHOLD:
                    continue
                overlap = min(fa, fb)
                if overlap < _MIN_OVERLAP_SCORE:
                    continue
                recs.append({
                    "stems": [a, b],
                    "type": "spectral_collision",
                    "band": band,
                    "band_hz": [lo, hi],
                    "score": round(overlap, 3),
                    "message": (
                        f"{STEM_LABELS.get(a, a)} y {STEM_LABELS.get(b, b)} compiten en la banda "
                        f"{band.replace('_', ' ')} ({lo}-{hi} Hz): {a} tiene "
                        f"{fa*100:.0f}% de su energía ahí y {b} tiene {fb*100:.0f}%. "
                        f"Sugerencia: EQ cut en uno de los dos alrededor de "
                        f"{(lo+hi)//2} Hz, o automatizar/side-chain para que no compitan al mismo tiempo."
                    ),
                })

    recs.sort(key=lambda r: r["score"], reverse=True)
    return recs


def _is_silent(stem_audio: np.ndarray) -> bool:
    mono = _to_mono(stem_audio)
    if mono.size == 0:
        return True
    rms = np.sqrt(np.mean(mono ** 2))
    return (20.0 * np.log10(rms + 1e-9)) < -55.0


def _check_kick_bass(mono_stems, sr, profiles, freqs_map, psd_map):
    kick_freq = _dominant_freq(freqs_map["drums"], psd_map["drums"], 35, 150)
    bass_freq = _dominant_freq(freqs_map["bass"], psd_map["bass"], 35, 250)
    if kick_freq <= 0 or bass_freq <= 0:
        return None

    env_kick = _low_band_envelope(mono_stems["drums"], sr, lo=30, hi=150)
    env_bass = _low_band_envelope(mono_stems["bass"], sr, lo=30, hi=150)
    corr = _envelope_correlation(env_kick, env_bass)

    freq_gap = abs(kick_freq - bass_freq)
    low_end_kick = profiles["drums"].get("low_end", 0.0) + profiles["drums"].get("sub", 0.0)
    low_end_bass = profiles["bass"].get("low_end", 0.0) + profiles["bass"].get("sub", 0.0)

    # Señal de colisión: fundamentales cercanos + ambos concentran energía
    # ahí + las envolventes se mueven juntas (pegan al mismo tiempo).
    is_problem = (freq_gap < 25.0) and (low_end_kick > 0.15) and (low_end_bass > 0.15) and (corr > 0.35)
    if not is_problem:
        return None

    score = round(min(1.0, (1.0 - freq_gap / 25.0) * 0.5 + max(0.0, corr) * 0.5), 3)
    lo, hi = int(min(kick_freq, bass_freq) - 15), int(max(kick_freq, bass_freq) + 15)

    return {
        "stems": ["drums", "bass"],
        "type": "kick_bass_collision",
        "band": "low_end",
        "band_hz": [max(20, lo), hi],
        "score": score,
        "fundamentals_hz": {"kick": round(kick_freq, 1), "bass": round(bass_freq, 1)},
        "envelope_correlation": round(corr, 2),
        "message": (
            f"🥁 El kick está tapando al bajo en ~{max(20, lo)}-{hi} Hz "
            f"(fundamental del kick ≈{kick_freq:.0f} Hz, del bajo ≈{bass_freq:.0f} Hz, "
            f"pegan casi al mismo tiempo — correlación de envolvente {corr:.2f}). "
            f"Sugerencias: side-chain del bajo disparado por el kick (attack rápido, "
            f"release 60-120ms), o separar fundamentales con EQ (ej. resonancia/boost "
            f"del kick en {kick_freq:.0f} Hz y corte del bajo ahí, dejándole al bajo "
            f"la zona por debajo de {min(kick_freq, bass_freq):.0f} Hz)."
        ),
    }


def analyze_stems_full(stems: dict, sr: int, lufs_fn=None) -> dict:
    """Punto de entrada único: análisis por stem + recomendaciones de masking."""
    per_stem = {
        name: analyze_stem(name, audio, sr, lufs_fn=lufs_fn)
        for name, audio in stems.items()
    }
    recommendations = detect_masking(stems, sr)
    return {
        "stems": per_stem,
        "recommendations": recommendations,
        "summary": (
            f"{len(recommendations)} colisión(es) detectada(s) entre stems."
            if recommendations else
            "No se detectaron colisiones espectrales significativas entre stems."
        ),
    }
