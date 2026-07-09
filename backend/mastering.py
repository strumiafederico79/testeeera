import dataclasses
import librosa
import soundfile as sf
import numpy as np
import uuid
import os
from scipy.signal import butter, sosfilt, sosfiltfilt, fftconvolve, resample_poly, welch, sosfreqz, firwin2, find_peaks
from scipy.ndimage import maximum_filter1d, median_filter

os.makedirs("processed", exist_ok=True)

# ─── MasteringParams: dataclass canónico de parámetros de la cadena ───────────
# MEJORA: centraliza los ~50 parámetros que antes se repetían literalmente en
# apply_mastering_chain, process_audio, y (los más críticos) en el bloque de
# LUFS safety check (donde duplicar manualmente cada kwarg era la fuente de
# bugs al agregar nuevos parámetros). Ahora todos los caminos usan dataclasses.asdict()
# y el safety check solo actualiza input_gain_db antes de re-renderizar.

@dataclasses.dataclass
class MasteringParams:
    # ── Ganancia / loudness ─────────────────────────────────────────────────
    input_gain_db: float = 0.0
    target_peak: float = 0.95        # aceptado pero no usado directamente en la cadena
    use_lufs_normalize: bool = False
    target_lufs: float = -14.0
    # ── Oversampling ────────────────────────────────────────────────────────
    oversample_mode: str = "quality"
    # ── High-pass ───────────────────────────────────────────────────────────
    hp_cutoff: float = 80.0
    # ── EQ paramétrica (4 bandas + high shelf) ──────────────────────────────
    eq_mode: str = "iir"
    linear_phase_taps: int = 2049
    high_shelf_gain_db: float = 0.0
    high_shelf_freq_hz: float = 8000.0
    eq1_freq: float = 100.0;  eq1_gain: float = 0.0;  eq1_q: float = 1.0
    eq2_freq: float = 500.0;  eq2_gain: float = 0.0;  eq2_q: float = 1.0
    eq3_freq: float = 2000.0; eq3_gain: float = 0.0;  eq3_q: float = 1.0
    eq4_freq: float = 8000.0; eq4_gain: float = 0.0;  eq4_q: float = 1.0
    # ── Dynamic EQ ──────────────────────────────────────────────────────────
    dyneq_bypass: bool = True
    dyneq_freq: float = 3000.0
    dyneq_q: float = 2.5
    dyneq_threshold_db: float = -18.0
    dyneq_ratio: float = 3.0
    dyneq_attack_ms: float = 3.0
    dyneq_release_ms: float = 80.0
    dyneq_max_reduction_db: float = 12.0
    # ── Transient shaper ────────────────────────────────────────────────────
    transient_attack: float = 0.0
    transient_sustain: float = 0.0
    # ── Compresor multibanda ─────────────────────────────────────────────────
    mb_bypass: bool = True
    mb_low_crossover: float = 250.0
    mb_high_crossover: float = 4000.0
    mb_low_threshold: float = 0.7;  mb_low_ratio: float = 2.0
    mb_low_attack_ms: float = 20.0; mb_low_release_ms: float = 150.0; mb_low_makeup_db: float = 0.0
    mb_mid_threshold: float = 0.7;  mb_mid_ratio: float = 2.0
    mb_mid_attack_ms: float = 20.0; mb_mid_release_ms: float = 150.0; mb_mid_makeup_db: float = 0.0
    mb_high_threshold: float = 0.7; mb_high_ratio: float = 2.0
    mb_high_attack_ms: float = 20.0; mb_high_release_ms: float = 150.0; mb_high_makeup_db: float = 0.0
    # ── Compresor de banda ancha ─────────────────────────────────────────────
    comp_stereo_link: bool = True
    comp_threshold: float = 0.5
    comp_ratio: float = 4.0
    comp_attack_ms: float = 10.0
    comp_release_ms: float = 100.0
    comp_makeup_db: float = 0.0
    # ── Glue compressor ──────────────────────────────────────────────────────
    glue_bypass: bool = True
    glue_threshold_db: float = -4.0
    glue_ratio: float = 2.0
    glue_attack_ms: float = 30.0
    glue_release_ms: float = 120.0
    glue_makeup_db: float = 0.0
    # ── Saturación armónica ──────────────────────────────────────────────────
    saturation_drive: float = 0.0
    saturation_mode: str = "tape"
    saturation_mix: float = 1.0
    # ── Imagen estéreo ───────────────────────────────────────────────────────
    mid_gain_db: float = 0.0
    side_gain_db: float = 0.0
    stereo_width_amount: float = 1.0
    use_stereo_enhancer: bool = False
    enhancer_bass_mono_freq: float = 120.0
    haas_delay_ms: float = 0.0
    low_end_mono_freq: float = 120.0
    low_end_mono_amount: float = 0.0
    mb_stereo_bypass: bool = True
    mb_stereo_low_width: float = 0.9
    mb_stereo_mid_width: float = 1.2
    mb_stereo_high_width: float = 1.5
    mb_stereo_low_crossover: float = 150.0
    mb_stereo_high_crossover: float = 4000.0
    # ── Reverb ───────────────────────────────────────────────────────────────
    reverb_size: float = 0.3
    reverb_wet: float = 0.0
    # ── Limitador ────────────────────────────────────────────────────────────
    limiter_ceiling: float = 0.95
    limiter_release_ms: float = 80.0
    # ── Reducción de ruido ───────────────────────────────────────────────────
    nr_bypass: bool = True
    nr_strength: float = 0.5
    nr_noise_sample_sec: float = 0.5

    def as_chain_kwargs(self) -> dict:
        """Devuelve todos los campos relevantes para apply_mastering_chain como dict."""
        return dataclasses.asdict(self)

    @classmethod
    def from_preset(cls, preset: dict) -> "MasteringParams":
        """Construye un MasteringParams a partir de un preset (ignora claves desconocidas)."""
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in preset.items() if k in valid})

def _report(cb, pct: float, stage: str) -> None:
    """Llama a `cb(pct, stage)` sin romper el procesamiento si el callback
    falla (por ejemplo si el job ya no existe o el dict fue limpiado)."""
    if cb is None:
        return
    try:
        cb(min(100, max(0, round(pct))), stage)
    except Exception:
        pass

DEFAULT_DSP_OVERSAMPLE = 4
OVERSAMPLING_MODES = {"off": 1, "draft": 1, "fast": 2, "quality": 4, "ultra": 8}

def resolve_oversample(mode: str | int | None = "quality") -> int:
    """Resolve oversampling quality names to integer factors."""
    if mode is None:
        return DEFAULT_DSP_OVERSAMPLE
    if isinstance(mode, (int, np.integer)):
        return max(1, int(mode))
    return OVERSAMPLING_MODES.get(str(mode).lower(), DEFAULT_DSP_OVERSAMPLE)


# ─── Numba acceleration ────────────────────────────────────────────────────────
try:
    import numba as nb
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# ─── Helpers ───────────────────────────────────────────────────────────────────

def _to_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return np.stack([audio, audio])
    if audio.shape[0] == 1:
        return np.concatenate([audio, audio], axis=0)
    return audio

def _crop_preview(audio: np.ndarray, sr: int, preview_seconds: float) -> np.ndarray:
    """Recorta un extracto de `preview_seconds` centrado en la MITAD del tema
    (en vez de los primeros N segundos). El arranque de un tema suele ser
    intro/silencio/poco representativo (drops, coros, secciones densas suelen
    estar en la mitad), así que un preview desde el segundo 0 tanto suena poco
    representativo para el oyente como, en el caso del matching por
    referencia, sesga el análisis espectral/dinámico usado para calcular el
    EQ de matching hacia una porción del tema que no representa el resto.
    """
    total_samples = audio.shape[1]
    max_samples = min(int(preview_seconds * sr), total_samples)
    if max_samples <= 0:
        return audio
    start = max(0, (total_samples - max_samples) // 2)
    return audio[:, start:start + max_samples]

# ─── Bit depth de salida + dithering ───────────────────────────────────────────
# Antes, sf.write() se llamaba sin `subtype`, así que soundfile caía en su
# default para WAV: PCM_16, truncando el float64 interno a 16 bits SIN
# dithering (libsndfile no ditherea por su cuenta). Truncar sin dither mete
# distorsión de cuantización correlacionada con la señal (audible como
# aspereza/artefactos en fades y pasajes de bajo nivel) en vez de un piso de
# ruido no correlacionado — es la diferencia entre un master "de estudio" y
# uno "de demo". Ahora la salida por defecto es 24-bit (headroom de sobra,
# nadie escucha el piso de cuantización a -144 dBFS) y, si el usuario elige
# igual bajar a 16-bit, se aplica dither TPDF antes de truncar.

OUTPUT_BIT_DEPTHS = {
    16: "PCM_16",
    24: "PCM_24",
    32: "FLOAT",   # 32-bit float: sin cuantización real, no necesita dither
}
DEFAULT_OUTPUT_BIT_DEPTH = 24

def resolve_bit_depth(bit_depth) -> int:
    try:
        bd = int(bit_depth)
    except (TypeError, ValueError):
        return DEFAULT_OUTPUT_BIT_DEPTH
    return bd if bd in OUTPUT_BIT_DEPTHS else DEFAULT_OUTPUT_BIT_DEPTH

def _tpdf_dither(audio: np.ndarray, bit_depth: int, seed: int = None) -> np.ndarray:
    """Dither TPDF (Triangular Probability Density Function), el estándar de
    facto en mastering para cuantizar de punto flotante a un bit depth entero
    (AES17 / práctica habitual de Sound Forge, RX, Ozone, etc.).

    Sumar dos variables uniformes independientes de amplitud ±0.5 LSB da una
    distribución triangular de ±1 LSB pico a pico. Esto decorrelaciona
    completamente el error de cuantización de la señal (a diferencia de
    rectangular dither, que todavía deja algo de correlación en señales de
    bajo nivel), a costa de un pelín más de piso de ruido — inaudible en la
    práctica e imperceptible frente al ruido térmico de cualquier conversor
    real.

    No hace nada para bit_depth >= 32 (punto flotante: no hay cuantización
    real que dithering).
    """
    if bit_depth >= 32:
        return audio
    rng = np.random.default_rng(seed)
    lsb = 2.0 / (2 ** bit_depth)  # rango [-1, 1] → 2.0 de span total
    noise = (rng.uniform(-0.5, 0.5, size=audio.shape) +
             rng.uniform(-0.5, 0.5, size=audio.shape)) * lsb
    return np.clip(audio + noise, -1.0, 1.0)

def _write_master_output(audio_out: np.ndarray, sr: int, output_path: str,
                         output_format: str, output_bit_depth: int = DEFAULT_OUTPUT_BIT_DEPTH,
                         dither_seed: int = None) -> int:
    """Escribe el archivo final aplicando dither TPDF si corresponde. Devuelve
    el bit depth efectivamente usado (para reportarlo en la respuesta de la
    API). Centraliza lo que antes estaba duplicado en process_audio() y
    process_audio_with_reference() para que ambos caminos escriban siempre
    igual y no se desincronicen con el próximo cambio.
    """
    bit_depth = resolve_bit_depth(output_bit_depth)
    subtype = OUTPUT_BIT_DEPTHS[bit_depth]
    dithered = _tpdf_dither(audio_out, bit_depth, seed=dither_seed)

    if output_format == "mp3":
        tmp_wav = output_path.replace(".mp3", "_tmp.wav")
        # El intermedio hacia MP3 se escribe en 24-bit SIN dither: el propio
        # encoder lossy va a volver a cuantizar/comprimir la señal, así que
        # ditherear acá no aporta nada y sólo sumaría ruido extra antes de la
        # compresión. El dither de bit depth tiene sentido en la entrega PCM
        # final (WAV/FLAC), que es la que realmente queda a ese bit depth.
        sf.write(tmp_wav, audio_out, sr, subtype="PCM_24")
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_wav(tmp_wav)
            seg.export(output_path, format="mp3", bitrate="320k")
        finally:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)
        return bit_depth

    sf.write(output_path, dithered, sr, subtype=subtype)
    return bit_depth

# ── Envelope follower (Numba JIT) ────────────────────────────────────────────────
if HAS_NUMBA:
    @nb.jit(nopython=True, cache=True, fastmath=True)
    def _smooth_envelope_numba(signal: np.ndarray, attack_coef: float, release_coef: float) -> np.ndarray:
        n = len(signal)
        env = np.empty(n, dtype=np.float64)
        prev = 0.0
        for i in range(n):
            x = signal[i]
            coef = attack_coef if x > prev else release_coef
            prev = coef * prev + (1.0 - coef) * x
            env[i] = prev
        return env

    @nb.jit(nopython=True, cache=True, fastmath=True)
    def _compute_gain_reduction_numba(env_db: np.ndarray, threshold_db: float, ratio: float,
                                      knee_db: float = 6.0) -> np.ndarray:
        # MEJORA: codo suave (soft-knee) en vez de codo duro. Con codo duro la
        # compresión arranca de golpe apenas se cruza el umbral (quiebre de
        # pendiente instantáneo en la curva de transferencia), lo que en
        # señales musicales se percibe como una compresión más brusca/audible
        # de lo necesario. La fórmula estándar (Zölzer/Giannoulis) hace una
        # transición cuadrática en una ventana de `knee_db` alrededor del
        # umbral, para que la ganancia entre suavemente. Fuera de esa ventana
        # el resultado es idéntico al codo duro de antes.
        n = len(env_db)
        gr = np.zeros(n, dtype=np.float64)
        half_knee = knee_db / 2.0
        factor = (1.0 / ratio) - 1.0
        for i in range(n):
            over = env_db[i] - threshold_db
            if knee_db > 0.0 and -half_knee < over < half_knee:
                gr[i] = factor * (over + half_knee) ** 2 / (2.0 * knee_db)
            elif over >= half_knee:
                gr[i] = factor * over
        return gr

    @nb.jit(nopython=True, cache=True, fastmath=True)
    def _limiter_gain_numba(instant_gain: np.ndarray, release_coef: float) -> np.ndarray:
        n = len(instant_gain)
        smoothed = np.empty(n, dtype=np.float64)
        prev = 1.0
        for i in range(n):
            g = instant_gain[i]
            if g < prev:
                prev = g
            else:
                prev = release_coef * prev + (1.0 - release_coef) * g
            smoothed[i] = prev
        return smoothed

def _soft_knee_gain_reduction_np(env_db: np.ndarray, threshold_db: float, ratio: float,
                                 knee_db: float = 6.0) -> np.ndarray:
    """Versión numpy (fallback sin numba) del soft-knee de _compute_gain_reduction_numba."""
    over = env_db - threshold_db
    half_knee = knee_db / 2.0
    factor = (1.0 / ratio) - 1.0
    hard = factor * over
    knee = factor * (over + half_knee) ** 2 / (2.0 * knee_db) if knee_db > 0.0 else hard
    gr = np.where(over >= half_knee, hard, np.where(over > -half_knee, knee, 0.0))
    return gr

def _smooth_envelope(signal: np.ndarray, sr: int, attack_ms: float, release_ms: float) -> np.ndarray:
    attack_coef  = np.exp(-1.0 / (sr * (attack_ms  / 1000.0) + 1e-9))
    release_coef = np.exp(-1.0 / (sr * (release_ms / 1000.0) + 1e-9))
    sig64 = signal.astype(np.float64, copy=False)
    if HAS_NUMBA:
        return _smooth_envelope_numba(sig64, attack_coef, release_coef)
    n = len(sig64)
    env = np.empty(n, dtype=np.float64)
    prev = 0.0
    for i in range(n):
        x = sig64[i]
        coef = attack_coef if x > prev else release_coef
        prev = coef * prev + (1.0 - coef) * x
        env[i] = prev
    return env

# ─── DSP primitives ────────────────────────────────────────────────────────────

def measure_lufs_integrated(audio: np.ndarray, sr: int) -> float:
    try:
        import pyloudnorm as pyln
        data = audio.T if audio.ndim == 2 else audio
        meter = pyln.Meter(sr)
        val = meter.integrated_loudness(data)
        if np.isfinite(val):
            return float(val)
    except Exception:
        pass
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    rms = np.sqrt(np.mean(mono ** 2)) + 1e-9
    return float(20.0 * np.log10(rms) - 0.691)

def eq_high_pass(audio: np.ndarray, sr: int, cutoff_hz: float = 80.0) -> np.ndarray:
    cutoff_hz = float(np.clip(cutoff_hz, 5.0, sr / 2.0 - 1.0))
    sos = butter(4, cutoff_hz, btype="highpass", fs=sr, output="sos")
    # Usar sosfiltfilt para fase cero y evitar artefactos
    if audio.ndim == 1:
        return sosfiltfilt(sos, audio)
    return np.stack([sosfiltfilt(sos, ch) for ch in audio])

def eq_high_shelf(audio: np.ndarray, sr: int,
                  cutoff_hz: float = 8000.0, gain_db: float = 2.0,
                  freq_hz: float = None) -> np.ndarray:
    # freq_hz overrides cutoff_hz when provided (alias for UI clarity)
    if freq_hz is not None:
        cutoff_hz = freq_hz
    if gain_db == 0.0:
        return audio
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * cutoff_hz / sr
    cos_w0, sin_w0 = np.cos(w0), np.sin(w0)
    alpha = sin_w0 / 2.0 * np.sqrt((A + 1.0 / A) * 1.0 + 2.0)  # BUGFIX: (1/1-1)=0 colapsaba la fórmula RBJ
    sqrtA = np.sqrt(A)
    b0 =  A * ((A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * sqrtA * alpha)
    b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
    b2 =  A * ((A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * sqrtA * alpha)
    a0 =       (A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * sqrtA * alpha
    a1 =  2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
    a2 =       (A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * sqrtA * alpha
    sos = np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])
    # BUGFIX: antes se usaba sosfilt (causal) acá mientras el resto de la EQ
    # (high-pass, bandas paramétricas) usa sosfiltfilt (fase cero). Esa
    # inconsistencia introducía un desfasaje relativo entre el high shelf y
    # las demás etapas de EQ, lo que al sumarse podía generar cancelaciones
    # sutiles/coloración no deseada. Ahora es fase cero, como el resto.
    if audio.ndim == 1:
        return sosfiltfilt(sos, audio)
    return np.stack([sosfiltfilt(sos, ch) for ch in audio])

def eq_parametric_band(audio: np.ndarray, sr: int,
                       freq: float, gain_db: float, q: float = 1.0) -> np.ndarray:
    if gain_db == 0.0:
        return audio
    freq = float(np.clip(freq, 20.0, sr / 2.0 - 1.0))
    q    = float(np.clip(q, 0.1, 30.0))
    A    = 10.0 ** (gain_db / 40.0)
    w0   = 2.0 * np.pi * freq / sr
    sin_w0 = np.sin(w0)
    cos_w0 = np.cos(w0)
    alpha = sin_w0 / (2.0 * q)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A
    sos = np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])
    if audio.ndim == 1:
        return sosfiltfilt(sos, audio)
    return np.stack([sosfiltfilt(sos, ch) for ch in audio])

def stereo_width(audio: np.ndarray, width: float = 1.2) -> np.ndarray:
    if audio.ndim != 2 or audio.shape[0] != 2:
        return audio
    mid  = (audio[0] + audio[1]) * 0.5
    side = (audio[0] - audio[1]) * 0.5 * width
    return np.stack([mid + side, mid - side])

def multiband_stereo_width(audio: np.ndarray, sr: int,
                           low_width: float = 0.8,
                           mid_width: float = 1.2,
                           high_width: float = 1.5,
                           low_crossover: float = 150.0,
                           high_crossover: float = 4000.0) -> np.ndarray:
    """Ancho estéreo multibanda: aplica un factor de width diferente a cada banda
    de frecuencia (graves, medios, agudos) usando filtros LP/HP de 2º orden.

    Ventajas vs. width global:
    - Los graves se pueden mantener casi mono (low_width≈0.8..1.0) para
      compatibilidad mono, potencia en sub y evitar cancelaciones de fase en
      sistemas mono/club.
    - Los medios y agudos pueden ensancharse independientemente para dar aire
      sin afectar la coherencia del bajo.

    low_width:  factor de ancho para la banda de graves (0=mono, 1=original, 2=doble)
    mid_width:  factor de ancho para la banda de medios
    high_width: factor de ancho para la banda de agudos
    low_crossover:  Hz de cruce graves→medios
    high_crossover: Hz de cruce medios→agudos
    """
    if audio.ndim != 2 or audio.shape[0] != 2:
        return audio
    low_crossover  = float(np.clip(low_crossover,  20.0, sr / 2.0 - 10.0))
    high_crossover = float(np.clip(high_crossover, low_crossover + 1.0, sr / 2.0 - 1.0))

    # BUGFIX: filtros Butterworth 2º orden LP+HP no suman a la señal original
    # en la zona de cruce (introducen coloración/phase notch). Se usan filtros
    # Linkwitz-Riley 4º orden (LR4 = cascada de dos Butterworth 2º orden con
    # sosfiltfilt, equivalente a cascadar dos veces) que sí suman perfectamente
    # en magnitud en el crossover — el estándar en crossovers de mastering.
    # sosfiltfilt ya aplica el filtro dos veces (forward+backward), así que
    # para LR4 basta con un butter(2) pasado por sosfiltfilt (2 pases × 2 polos
    # = 4 polos efectivos, -24 dB/oct, suma perfecta).
    sos_lo_lp = butter(2, low_crossover,  btype='lowpass',  fs=sr, output='sos')
    sos_lo_hp = butter(2, low_crossover,  btype='highpass', fs=sr, output='sos')
    sos_hi_lp = butter(2, high_crossover, btype='lowpass',  fs=sr, output='sos')
    sos_hi_hp = butter(2, high_crossover, btype='highpass', fs=sr, output='sos')

    # Aplicar dos veces cada filtro (forward+backward) para LR4
    def _lr4(sos, x):
        return sosfiltfilt(sos, sosfiltfilt(sos, x))

    low_band  = _lr4(sos_lo_lp, audio)
    mh_band   = _lr4(sos_lo_hp, audio)
    mid_band  = _lr4(sos_hi_lp, mh_band)
    high_band = _lr4(sos_hi_hp, mh_band)

    def _apply_width(band, w):
        mid  = (band[0] + band[1]) * 0.5
        side = (band[0] - band[1]) * 0.5 * w
        return np.stack([mid + side, mid - side])

    return _apply_width(low_band, low_width) + _apply_width(mid_band, mid_width) + _apply_width(high_band, high_width)

def reverb_simple(audio: np.ndarray, sr: int,
                  room_size: float = 0.3, wet: float = 0.1,
                  seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    decay_samples = max(int(sr * room_size), 1)
    t  = np.linspace(0.0, room_size, decay_samples)
    ir = np.exp(-6.0 * t) * rng.standard_normal(decay_samples) * 0.5
    ir[0] = 1.0
    dry = 1.0 - wet
    if audio.ndim == 1:
        wet_sig = fftconvolve(audio, ir, mode="full")[:len(audio)]
        return dry * audio + wet * wet_sig
    out = []
    for ch in audio:
        wet_sig = fftconvolve(ch, ir, mode="full")[:len(ch)]
        out.append(dry * ch + wet * wet_sig)
    return np.stack(out)

def limiter(audio: np.ndarray, sr: int,
            ceiling: float = 0.95, release_ms: float = 80.0,
            lookahead_ms: float = 5.0, oversample: int = DEFAULT_DSP_OVERSAMPLE) -> np.ndarray:
    """True Peak limiter brick-wall con lookahead, ganancia suavizada, oversampling
    y detección LINKEADA entre canales (estéreo).

    El oversampling (por defecto 4x) permite detectar los inter-sample peaks
    (picos que aparecen entre samples al reconstruir la señal analógica) que un
    limiter a sample-rate nativa no ve — el estándar AES17 / EBU R128 llama a
    esto "True Peak". Sin oversampling un -0.1 dBFS digital puede reproducirse
    en analógico como +0.5 dBFS o más, causando clipping en conversores DAC.
    Con oversampling 4x el margen de error se reduce a <0.01 dB típicamente.

    BUGFIX (corrimiento de imagen estéreo): antes cada canal calculaba su
    propia curva de ganancia de forma independiente. Si L pedía más
    reducción que R en un instante (p.ej. un platillo panneado a la
    izquierda), L se atenuaba más que R en ESE instante, moviendo el
    balance L/R momentáneamente — un limiter "unlinked" corre/empuja la
    imagen estéreo con la música. Un limiter de máster debe estar linkeado:
    se detecta el pico máximo entre canales y se aplica LA MISMA curva de
    ganancia a ambos, preservando el panorama exactamente como estaba.
    """
    release_coef = float(np.exp(-1.0 / (sr * (release_ms / 1000.0) + 1e-9)))
    lookahead_samples = max(1, int(sr * lookahead_ms / 1000.0))
    ovs = max(1, int(oversample))

    def gain_curve(abs_signal):
        if ovs > 1:
            abs_up = resample_poly(abs_signal, ovs, 1)
            abs_up = np.abs(abs_up)
            la_up = max(1, lookahead_samples * ovs)
            if la_up > 1:
                fwd_max = maximum_filter1d(abs_up, size=la_up, mode='nearest')
                # BUGFIX: np.roll es circular — las muestras del final volvían
                # al principio generando reducción de ganancia fantasma en el
                # tail del archivo. Se usa desplazamiento lineal con cola = 0
                # (sin pico → ganancia 1.0, no wrap).
                shift = la_up // 2
                abs_up_la = np.empty_like(fwd_max)
                abs_up_la[:len(fwd_max) - shift] = fwd_max[shift:]
                abs_up_la[len(fwd_max) - shift:] = 0.0
            else:
                abs_up_la = abs_up
            ig_up = np.minimum(1.0, ceiling / (abs_up_la + 1e-9)).astype(np.float64)
            if HAS_NUMBA:
                smoothed_up = _limiter_gain_numba(ig_up, float(np.exp(-1.0 / (sr * ovs * (release_ms / 1000.0) + 1e-9))))
            else:
                rc = float(np.exp(-1.0 / (sr * ovs * (release_ms / 1000.0) + 1e-9)))
                n = len(ig_up)
                smoothed_up = np.empty(n, dtype=np.float64)
                prev = 1.0
                for i in range(n):
                    g = ig_up[i]
                    prev = g if g < prev else rc * prev + (1.0 - rc) * g
                    smoothed_up[i] = prev
            smoothed = resample_poly(smoothed_up, 1, ovs)[:len(abs_signal)]
            return np.clip(smoothed, 0.0, 1.0)
        else:
            abs_ch = np.abs(abs_signal)
            if lookahead_samples > 1:
                fwd_max = maximum_filter1d(abs_ch, size=lookahead_samples, mode='nearest')
                # BUGFIX: mismo fix que el path oversampled — desplazamiento
                # lineal sin wrap circular para evitar artefactos en el tail.
                shift = lookahead_samples // 2
                abs_ch_la = np.empty_like(fwd_max)
                abs_ch_la[:len(fwd_max) - shift] = fwd_max[shift:]
                abs_ch_la[len(fwd_max) - shift:] = 0.0
            else:
                abs_ch_la = abs_ch
            ig = np.minimum(1.0, ceiling / (abs_ch_la + 1e-9)).astype(np.float64)
            if HAS_NUMBA:
                return _limiter_gain_numba(ig, release_coef)
            n = len(ig)
            smoothed = np.empty(n, dtype=np.float64)
            prev = 1.0
            for i in range(n):
                g = ig[i]
                prev = g if g < prev else release_coef * prev + (1.0 - release_coef) * g
                smoothed[i] = prev
            return smoothed

    if audio.ndim == 1:
        smoothed = gain_curve(audio)
        out = audio * smoothed
    else:
        # Detección linkeada: se toma el máximo absoluto ENTRE canales en
        # cada instante (equivalente a "peak between L/R" de un limiter
        # estéreo real) y esa única curva de ganancia se aplica a todos los
        # canales por igual, preservando el panorama.
        linked_peak = np.max(np.abs(audio), axis=0)
        smoothed = gain_curve(linked_peak)
        out = audio * smoothed[np.newaxis, :]
    return out

def mid_side_process(audio: np.ndarray,
                     mid_gain_db: float = 0.0,
                     side_gain_db: float = 0.0) -> np.ndarray:
    if audio.ndim != 2 or audio.shape[0] != 2:
        return audio
    left, right = audio[0], audio[1]
    mid  = (left + right) * 0.5
    side = (left - right) * 0.5
    mid  *= 10.0 ** (mid_gain_db  / 20.0)
    side *= 10.0 ** (side_gain_db / 20.0)
    return np.stack([mid + side, mid - side])

def low_end_mono_maker(audio: np.ndarray, sr: int,
                       freq: float = 120.0, mono_amount: float = 1.0) -> np.ndarray:
    """Mono Maker de graves DEDICADO (independiente del stereo_enhancer, que
    trae su propio bass-mono fijo). Por debajo de `freq` reduce el ancho
    estéreo hacia mono en la proporción `mono_amount` (0 = estéreo intacto,
    1 = mono total); por encima de `freq` no toca nada. Usa crossover
    Butterworth 4º orden en fase cero (sosfiltfilt), igual convención que el
    resto de la EQ. Sirve para compatibilidad mono, evitar cancelaciones de
    fase de graves en sistemas club/vinilo, y concentrar energía de sub en
    el centro — el pedido típico de "graves en mono <100-150 Hz".
    """
    if audio.ndim != 2 or audio.shape[0] != 2 or mono_amount <= 0.0:
        return audio
    freq = float(np.clip(freq, 20.0, sr / 2.0 - 1.0))
    mono_amount = float(np.clip(mono_amount, 0.0, 1.0))
    sos_lp = butter(4, freq, btype="lowpass",  fs=sr, output="sos")
    sos_hp = butter(4, freq, btype="highpass", fs=sr, output="sos")
    low  = sosfiltfilt(sos_lp, audio)
    high = sosfiltfilt(sos_hp, audio)
    low_mono = np.tile(low.mean(axis=0), (2, 1))
    low_out  = low * (1.0 - mono_amount) + low_mono * mono_amount
    return low_out + high

# ─── Diseño de filtros RBJ (biquad) reutilizables como SOS ────────────────────
# Extraídos como helpers independientes para poder evaluar su respuesta en
# frecuencia (sosfreqz) y sumarla en dB al diseñar el EQ de fase lineal FIR
# más abajo, sin duplicar/desincronizar las fórmulas de eq_parametric_band /
# eq_high_shelf.

def _design_peaking_sos(sr: int, freq: float, gain_db: float, q: float) -> np.ndarray:
    freq = float(np.clip(freq, 20.0, sr / 2.0 - 1.0))
    q    = float(np.clip(q, 0.1, 30.0))
    A    = 10.0 ** (gain_db / 40.0)
    w0   = 2.0 * np.pi * freq / sr
    cos_w0, sin_w0 = np.cos(w0), np.sin(w0)
    alpha = sin_w0 / (2.0 * q)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])

def _design_high_shelf_sos(sr: int, freq: float, gain_db: float) -> np.ndarray:
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq / sr
    cos_w0, sin_w0 = np.cos(w0), np.sin(w0)
    alpha = sin_w0 / 2.0 * np.sqrt((A + 1.0 / A) * 1.0 + 2.0)
    sqrtA = np.sqrt(A)
    b0 =  A * ((A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * sqrtA * alpha)
    b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
    b2 =  A * ((A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * sqrtA * alpha)
    a0 =       (A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * sqrtA * alpha
    a1 =  2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
    a2 =       (A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * sqrtA * alpha
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])

def _design_low_shelf_sos(sr: int, freq: float, gain_db: float) -> np.ndarray:
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq / sr
    cos_w0, sin_w0 = np.cos(w0), np.sin(w0)
    alpha = sin_w0 / 2.0 * np.sqrt((A + 1.0 / A) * 1.0 + 2.0)
    sqrtA = np.sqrt(A)
    b0 =    A * ((A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * sqrtA * alpha)
    b1 =  2.0 * A * ((A - 1.0) - (A + 1.0) * cos_w0)
    b2 =    A * ((A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * sqrtA * alpha)
    a0 =         (A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * sqrtA * alpha
    a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cos_w0)
    a2 =         (A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * sqrtA * alpha
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])

def linear_phase_eq(audio: np.ndarray, sr: int, bands: list, num_taps: int = 2049) -> np.ndarray:
    """EQ de FASE LINEAL real (FIR), a diferencia de eq_parametric_band /
    eq_high_shelf que son IIR con sosfiltfilt (fase CERO, no lineal: fase
    cero es más fuerte que lineal pero exige procesar offline con la señal
    completa, cosa que igual hacemos acá, así que en la práctica ya nos daba
    fase cero). El motivo real para tener un modo FIR de fase lineal
    dedicado es tener delay de grupo CONSTANTE y verificable en todas las
    frecuencias (como en plugins tipo Pro-Q "Linear Phase"), útil cuando se
    van a sumar/comparar bandas o cuando el pre-ringing controlado del FIR
    es preferible a la respuesta IIR en cortes/boosts grandes.

    bands: lista de dicts, cada uno:
        {"type": "peak", "freq": 100.0, "gain_db": 2.0, "q": 1.0}
        {"type": "high_shelf", "freq": 8000.0, "gain_db": 2.0}
        {"type": "low_shelf",  "freq": 100.0,  "gain_db": 2.0}
    Se combinan TODAS las bandas en una sola curva de magnitud (suma en dB),
    y de ahí se diseña UN solo FIR (firwin2) — evita cascadear N filtros y
    acumular N delays/errores de diseño por separado.
    """
    bands = [b for b in (bands or []) if b.get("gain_db", 0.0) != 0.0]
    if not bands:
        return audio
    num_taps = int(num_taps)
    if num_taps % 2 == 0:
        num_taps += 1  # taps impares -> FIR simétrico Tipo I, fase lineal exacta

    n_freqs = 4096
    freqs_grid = np.linspace(0.0, sr / 2.0, n_freqs)
    total_db = np.zeros(n_freqs)
    for b in bands:
        gain_db = float(b.get("gain_db", 0.0))
        freq = float(np.clip(b.get("freq", 1000.0), 20.0, sr / 2.0 - 1.0))
        btype = b.get("type", "peak")
        if btype == "high_shelf":
            sos = _design_high_shelf_sos(sr, freq, gain_db)
        elif btype == "low_shelf":
            sos = _design_low_shelf_sos(sr, freq, gain_db)
        else:
            q = float(np.clip(b.get("q", 1.0), 0.1, 30.0))
            sos = _design_peaking_sos(sr, freq, gain_db, q)
        _, h = sosfreqz(sos, worN=freqs_grid, fs=sr)
        total_db += 20.0 * np.log10(np.abs(h) + 1e-12)

    gain_lin = 10.0 ** (total_db / 20.0)
    freq_norm = freqs_grid / (sr / 2.0)
    freq_norm[-1] = 1.0
    taps = firwin2(num_taps, freq_norm, gain_lin)

    def _apply(ch):
        # fftconvolve + mode='same' con FIR simétrico (impar) cancela el
        # delay de grupo (num_taps-1)/2: el tap central queda alineado con
        # t=0, dando salida sin corrimiento temporal audible.
        return fftconvolve(ch, taps, mode='same')

    if audio.ndim == 1:
        return _apply(audio)
    return np.stack([_apply(ch) for ch in audio])

def detect_resonances(audio: np.ndarray, sr: int,
                      min_freq: float = 120.0, max_freq: float = 9000.0,
                      threshold_db: float = 4.0, max_resonances: int = 6) -> list:
    """Detección de resonancias: busca picos ANGOSTOS que sobresalen del
    perfil espectral general (baseline = mediana móvil en el propio
    espectro promediado por Welch), no simplemente "la banda más alta" —
    eso sería balance tonal, no resonancia. Cada resultado trae una
    sugerencia de corte (Dynamic EQ / EQ estático) lista para aplicar con
    dynamic_eq_band(freq=r['freq_hz'], q=r['suggested_q'], ...).
    """
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    n_fft = 8192
    mag = _averaged_magnitude_spectrum(mono, n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    mag_db = 20.0 * np.log10(mag + 1e-12)

    band_mask = (freqs >= min_freq) & (freqs <= max_freq)
    idx = np.where(band_mask)[0]
    if len(idx) < 10:
        return []

    win = max(5, (int(len(idx) * 0.03) | 1))  # ventana impar, ~3% del rango
    baseline = median_filter(mag_db[idx], size=win)
    excess = mag_db[idx] - baseline

    peaks, props = find_peaks(excess, height=threshold_db, distance=max(3, win // 2))
    results = []
    for p, h in zip(peaks, props["peak_heights"]):
        f = float(freqs[idx][p])
        results.append({
            "freq_hz": round(f, 1),
            "excess_db": round(float(h), 2),
            "suggested_cut_db": round(float(min(h * 0.7, 6.0)), 2),
            "suggested_q": 4.0,
        })
    results.sort(key=lambda r: -r["excess_db"])
    return results[:max_resonances]

def detect_sibilance(audio: np.ndarray, sr: int,
                     low_hz: float = 4000.0, high_hz: float = 9000.0,
                     block_s: float = 0.05) -> dict:
    """Detección de sibilancia: compara la envolvente de energía de la
    banda 4-9kHz contra la envolvente de banda completa, cuadro a cuadro.
    No basta con "hay mucha energía en agudos" (eso es balance tonal) — lo
    que caracteriza la sibilancia son PICOS puntuales de esa banda por
    encima de su propia mediana (las "eses"/"ches" sobresaliendo del resto
    de la voz). suggested_reduction_db queda pensado para alimentar
    dynamic_eq_band como de-esser (freq≈centro de la banda, Q ancho).
    """
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    high_hz = min(high_hz, sr / 2.0 - 1.0)
    if high_hz <= low_hz:
        return {"present": False, "band_hz": [low_hz, high_hz], "severity_db": 0.0,
                "frames_flagged_pct": 0.0, "suggested_reduction_db": 0.0}

    band = _bandpass_filter(mono, sr, low_hz, high_hz)
    hop = max(1, int(sr * block_s))
    n_blocks = max(1, len(mono) // hop)
    band_env_db, full_env_db = [], []
    for i in range(n_blocks):
        seg_b = band[i * hop:(i + 1) * hop]
        seg_f = mono[i * hop:(i + 1) * hop]
        if len(seg_b) == 0:
            continue
        band_env_db.append(20.0 * np.log10(np.sqrt(np.mean(seg_b ** 2)) + 1e-9))
        full_env_db.append(20.0 * np.log10(np.sqrt(np.mean(seg_f ** 2)) + 1e-9))
    if not band_env_db:
        return {"present": False, "band_hz": [low_hz, high_hz], "severity_db": 0.0,
                "frames_flagged_pct": 0.0, "suggested_reduction_db": 0.0}

    ratio_db = np.array(band_env_db) - np.array(full_env_db)
    baseline = float(np.median(ratio_db))
    spikes = ratio_db - baseline
    flagged = spikes > 6.0
    severity = float(np.mean(spikes[spikes > 0])) if np.any(spikes > 0) else 0.0
    present = bool(np.mean(flagged) > 0.03 and severity > 3.0)

    return {
        "present": present,
        "band_hz": [round(low_hz, 1), round(high_hz, 1)],
        "severity_db": round(severity, 2),
        "frames_flagged_pct": round(float(np.mean(flagged) * 100.0), 1),
        "suggested_reduction_db": round(float(min(severity, 8.0)), 2),
    }

def dynamic_eq_band(audio: np.ndarray, sr: int,
                    freq: float = 3000.0, q: float = 2.5,
                    threshold_db: float = -18.0, ratio: float = 3.0,
                    attack_ms: float = 3.0, release_ms: float = 80.0,
                    max_reduction_db: float = 12.0, bypass: bool = True) -> tuple:
    """Dynamic EQ de banda única: aísla una banda angosta (freq/Q) con un
    bandpass de fase cero, comprime HACIA ABAJO solo esa banda cuando su
    envolvente supera threshold_db, y la recombina con el resto de la señal
    intacto (residual = señal - banda; salida = residual + banda_comprimida).

    Es un bloque GENÉRICO: con Q alto en la frecuencia de una resonancia
    detectada actúa como de-resonador; con freq≈6500Hz y Q ancho actúa como
    de-esser. No son dos herramientas separadas, son el mismo mecanismo
    apuntado a distintas zonas — así lo pedía el ítem "Dynamic EQ en
    algunas zonas" + "reducción de resonancias" + "detección de
    sibilancias" del pedido.
    """
    if bypass or max_reduction_db <= 0.0:
        return audio, {"bypass": True, "gr_db": 0.0}

    freq = float(np.clip(freq, 20.0, sr / 2.0 - 1.0))
    q = float(np.clip(q, 0.3, 30.0))
    bw = freq / q
    lo = max(20.0, freq - bw / 2.0)
    hi = min(sr / 2.0 - 1.0, freq + bw / 2.0)
    if hi <= lo:
        return audio, {"bypass": True, "gr_db": 0.0}

    def process_channel(ch):
        band = _bandpass_filter(ch, sr, lo, hi)
        residual = ch - band
        env = _smooth_envelope(np.abs(band), sr, attack_ms, release_ms)
        env_db = 20.0 * np.log10(env + 1e-9)
        gr_db = _soft_knee_gain_reduction_np(env_db, threshold_db, ratio)
        gr_db = np.maximum(gr_db, -max_reduction_db)
        band_out = band * (10.0 ** (gr_db / 20.0))
        return residual + band_out, gr_db

    if audio.ndim == 1:
        out, gr_db = process_channel(audio)
    else:
        outs, grs = [], []
        for ch in audio:
            o, g = process_channel(ch)
            outs.append(o)
            grs.append(g)
        out = np.stack(outs)
        gr_db = np.mean(np.stack(grs), axis=0)

    tail = max(1, sr // 8)
    meter = {
        "bypass": False,
        "freq_hz": round(freq, 1),
        "q": round(q, 2),
        "band_range_hz": [round(lo, 1), round(hi, 1)],
        "gr_db": round(float(np.mean(gr_db[-tail:])), 2),
    }
    return out, meter

def transient_shaper(audio: np.ndarray, sr: int,
                     attack_amount: float = 0.0, sustain_amount: float = 0.0,
                     attack_time_ms: float = 5.0, release_time_ms: float = 80.0) -> np.ndarray:
    if attack_amount == 0.0 and sustain_amount == 0.0:
        return audio

    def process_channel(ch):
        abs_ch = np.abs(ch)
        fast_env = _smooth_envelope(abs_ch, sr, attack_time_ms, attack_time_ms * 2.0)
        slow_env = _smooth_envelope(abs_ch, sr, release_time_ms, release_time_ms)
        transient_comp = np.maximum(fast_env - slow_env, 0.0)
        sustain_comp   = np.minimum(fast_env, slow_env)
        denom = transient_comp + sustain_comp + 1e-9
        attack_gain  = 1.0 + attack_amount  * (transient_comp / denom)
        sustain_gain = 1.0 + sustain_amount * (sustain_comp   / denom)
        return ch * attack_gain * sustain_gain

    if audio.ndim == 1:
        return process_channel(audio)
    return np.stack([process_channel(ch) for ch in audio])

def harmonic_saturation(audio: np.ndarray,
                        drive: float = 0.2, mode: str = "tape",
                        mix: float = 1.0, oversample: int = DEFAULT_DSP_OVERSAMPLE) -> np.ndarray:
    """Saturación armónica (tape/tube) con waveshaper tanh.

    BUGFIX (saturación "extremadamente fuerte/distorsionada"):
    1) k = 1 + drive*10 era demasiado empinado: con drive=0.1 el RMS casi se
       duplicaba y el pico casi tocaba el techo. Ahora k = 1 + drive*5 (rango
       1x..6x), mucho más manejable.
    2) Se normalizaba dividiendo por tanh(k), lo que empuja la señal hacia
       ±1 para CUALQUIER drive > ~0.1, sin importar el nivel de entrada (esto
       es lo que hacía que sonara siempre "al borde del clip"). Ahora se
       normaliza por 'k' (ganancia de pequeña señal: d/dx tanh(kx)|x=0 = k),
       lo que preserva el nivel de la señal y satura solo los picos/transientes,
       comportándose como un saturador real en vez de un limitador disfrazado.
    3) Se agrega oversampling 4x (por defecto) antes del waveshaper no-lineal
       para reducir aliasing/artefactos digitales ásperos, típicos de aplicar
       tanh directamente a la frecuencia de muestreo original.
    """
    if drive <= 0.0 or mix <= 0.0:
        return audio
    k = 1.0 + drive * 5.0

    def shape(x):
        if mode == "tube":
            driven = x * k
            t = np.tanh(driven)
            wet = t - 0.15 * (t ** 2) * np.sign(driven)
        else:
            wet = np.tanh(x * k)
        return wet / k  # normaliza por ganancia de pequeña señal, no por tanh(k)

    def process_channel(ch):
        if oversample and oversample > 1:
            up = resample_poly(ch, oversample, 1)
            wet_up = shape(up)
            wet = resample_poly(wet_up, 1, oversample)[:len(ch)]
        else:
            wet = shape(ch)
        return (1.0 - mix) * ch + mix * wet

    if audio.ndim == 1:
        return process_channel(audio)
    return np.stack([process_channel(ch) for ch in audio])

def noise_reduction(
    audio: np.ndarray,
    sr: int,
    strength: float = 0.5,
    noise_sample_sec: float = 0.5,
) -> np.ndarray:
    """Reducción de ruido espectral (hiss, hum, ruido de sala).

    Estima el perfil de ruido a partir de los primeros `noise_sample_sec`
    segundos del track (asumiendo que ahí hay silencio o ruido de fondo),
    luego aplica sustracción espectral suavizada canal por canal.

    Parámetros:
        strength          : 0.0 = sin reducción, 1.0 = máxima agresividad.
        noise_sample_sec  : duración (s) de la zona de muestra de ruido.
                           0.0 usa la primera ventana disponible.

    Diseño:
    - Se usa noisereduce (biblioteca especializada) con prop_decrease=strength.
    - Si noisereduce no está instalado, hace una sustracción espectral manual
      con scipy (fallback sin dependencia extra).
    - Opera canal por canal para no mezclar el perfil de ruido L/R.
    - Preserva exactamente la longitud de la señal de entrada.
    """
    if strength <= 0.0:
        return audio

    strength = float(np.clip(strength, 0.0, 1.0))
    noise_samples = max(1, int(noise_sample_sec * sr))

    def _reduce_channel(ch: np.ndarray) -> np.ndarray:
        try:
            import noisereduce as nr
            noise_clip = ch[:noise_samples]
            return nr.reduce_noise(
                y=ch,
                y_noise=noise_clip,
                sr=sr,
                prop_decrease=strength,
                stationary=False,
                n_std_thresh_stationary=1.5,
            ).astype(np.float32)
        except ImportError:
            # Fallback: sustracción espectral manual con scipy
            from scipy.signal import stft, istft
            n_fft = 2048
            hop = n_fft // 4
            _, _, Zxx = stft(ch, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
            noise_profile = np.mean(np.abs(Zxx[:, :max(1, noise_samples // hop)]), axis=1, keepdims=True)
            mag = np.abs(Zxx)
            phase = np.angle(Zxx)
            mag_reduced = np.maximum(mag - noise_profile * strength, 0.0)
            _, recovered = istft(mag_reduced * np.exp(1j * phase), fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
            # Recortar o padear para preservar longitud exacta
            if len(recovered) >= len(ch):
                return recovered[:len(ch)].astype(np.float32)
            pad = np.zeros(len(ch) - len(recovered), dtype=np.float32)
            return np.concatenate([recovered, pad]).astype(np.float32)

    if audio.ndim == 1:
        return _reduce_channel(audio)
    return np.stack([_reduce_channel(ch) for ch in audio])


def stereo_enhancer(audio: np.ndarray, sr: int,
                    width: float = 1.3, bass_mono_freq: float = 120.0,
                    haas_delay_ms: float = 0.0) -> np.ndarray:
    if audio.ndim != 2 or audio.shape[0] != 2:
        return audio
    bass_mono_freq = float(np.clip(bass_mono_freq, 20.0, sr / 2.0 - 1.0))
    sos_lp = butter(2, bass_mono_freq, btype="lowpass",  fs=sr, output="sos")
    sos_hp = butter(2, bass_mono_freq, btype="highpass", fs=sr, output="sos")
    left, right = audio[0], audio[1]
    mono_sum = (left + right) * 0.5
    bass = np.stack([sosfiltfilt(sos_lp, mono_sum), sosfiltfilt(sos_lp, mono_sum)])
    high_l = sosfiltfilt(sos_hp, left)
    high_r = sosfiltfilt(sos_hp, right)
    mid  = (high_l + high_r) * 0.5
    side = (high_l - high_r) * 0.5 * width
    if haas_delay_ms > 0.0:
        delay_samples = max(1, int(sr * haas_delay_ms / 1000.0))
        side = np.concatenate([np.zeros(delay_samples), side])[:len(side)]
    highs = np.stack([mid + side, mid - side])
    return bass + highs

def _averaged_magnitude_spectrum(mono: np.ndarray, n_fft: int, max_frames: int = 2000) -> np.ndarray:
    """Espectro de magnitud promediado (método de Welch) sobre TODA la señal,
    no solo un puñado de frames sueltos.

    BUGFIX importante: la versión anterior tomaba `max_frames` frames (32 a 64)
    de n_fft muestras cada uno —apenas 1 a 6 segundos reales— y promediaba
    SOLO esos puntos, aunque estuvieran "repartidos" a lo largo del archivo.
    Para un tema de 3-4 minutos eso es <2% del audio: si algún frame caía
    justo sobre un silencio, una pausa vocal o un golpe de bombo, el balance
    espectral resultante podía salir completamente distinto entre dos
    análisis del mismo tema (o entre tema y referencia), y eso es lo que se
    veía como "el análisis da cualquier resultado". Welch promedia la energía
    de TODOS los frames que entran en la señal (con ventana Hann y 75% de
    solapamiento), así que el resultado usa el 100% del audio y es estable
    y repetible. `max_frames` ahora solo actúa como límite de cómputo para
    archivos larguísimos (agranda el hop, pero sigue cubriendo todo el
    archivo de punta a punta en vez de recortarlo).
    """
    n = len(mono)
    nperseg = int(min(n_fft, max(8, n)))
    if n <= nperseg:
        window = np.hanning(nperseg)
        frame = np.zeros(nperseg)
        frame[:n] = mono[:n]
        mag = np.abs(np.fft.rfft(frame * window))
        if nperseg < n_fft:
            target_len = n_fft // 2 + 1
            mag = np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(mag)), mag)
        return mag
    hop = max(1, nperseg // 4)
    n_segments = (n - nperseg) // hop + 1
    if n_segments > max_frames:
        hop = max(nperseg // 4, (n - nperseg) // max_frames + 1)
    noverlap = max(0, nperseg - hop)
    _freqs, psd = welch(mono, fs=1.0, window="hann", nperseg=nperseg,
                         noverlap=noverlap, scaling="spectrum", detrend=False,
                         average="mean")
    return np.sqrt(np.maximum(psd, 0.0))


def _log_band_average_db(freqs: np.ndarray, mag: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Promedia la magnitud del espectro (en energía) dentro de cada banda
    logarítmica definida por `edges`, devolviendo el resultado en dB.

    BUGFIX importante: con binning log puro, las bandas graves (p.ej. entre
    20 y 40Hz) suelen ser más angostas que la resolución real de la FFT
    (a 44.1kHz/4096 puntos, cada bin de FFT son ~10.8Hz). Eso hacía que
    varias bandas de salida no tuvieran NINGÚN bin de FFT adentro, y el
    código anterior les asignaba un piso arbitrario de -180dB — lo que se
    veía como dientes de sierra sin sentido en el extremo grave del gráfico
    (exactamente el "espectro que es cualquier cosa" / falta de una curva
    de respuesta en frecuencia coherente). Ahora, si una banda no tiene
    ningún bin de FFT adentro, se interpola en log-frecuencia a partir de
    los bins vecinos reales, en vez de inventar un valor.
    """
    log_freqs = np.log10(np.maximum(freqs, 1e-6))
    mag_db_full = 20.0 * np.log10(mag + 1e-9)
    out = np.empty(len(edges) - 1)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (freqs >= lo) & (freqs < hi)
        if mask.any():
            energy = float(np.sqrt(np.mean(mag[mask] ** 2)))
            out[i] = 20.0 * np.log10(energy + 1e-9)
        else:
            center = 0.5 * (lo + hi)
            out[i] = float(np.interp(np.log10(max(center, 1e-6)), log_freqs, mag_db_full))
    return out

def spectrum_analysis_fft(audio: np.ndarray, sr: int,
                          n_fft: int = 4096, n_bins: int = 64) -> dict:
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    if len(mono) < n_fft:
        n_fft = max(64, 2 ** int(np.floor(np.log2(max(len(mono), 64)))))
    avg_mag = _averaged_magnitude_spectrum(mono, n_fft)
    freqs    = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    nyquist  = sr / 2.0
    log_edges = np.logspace(np.log10(20.0), np.log10(nyquist), n_bins + 1)
    bin_db   = _log_band_average_db(freqs, avg_mag, log_edges).round(2).tolist()
    bin_freq = [round(float((log_edges[i] + log_edges[i + 1]) * 0.5), 1) for i in range(n_bins)]
    return {"frequencies_hz": bin_freq, "magnitudes_db": bin_db, "n_fft": n_fft}

# ─── Análisis ──────────────────────────────────────────────────────────────────

def stereo_correlation(audio: np.ndarray) -> float:
    if audio.ndim != 2 or audio.shape[0] != 2:
        return 1.0
    l, r = audio[0], audio[1]
    std_l, std_r = np.std(l), np.std(r)
    if std_l < 1e-9 or std_r < 1e-9:
        return 1.0
    corr = float(np.mean((l - l.mean()) * (r - r.mean())) / (std_l * std_r))
    return float(np.clip(corr, -1.0, 1.0))

def measure_lra(audio: np.ndarray, sr: int, block_s: float = 3.0, hop_s: float = 1.0) -> float:
    """Loudness Range (LRA) simplificado, estilo EBU R128: RMS por ventanas
    deslizantes (bloques de 3s, hop 1s) en dB, con gate absoluto (-70 dB) y
    gate relativo (descarta bloques 20 dB por debajo de la media) antes de
    tomar el rango entre percentiles 10 y 95. No es una implementación
    100% conforme al estándar (no usa K-weighting ni el gate de 2 pasadas
    exacto), pero es una proxy robusta y estable para comparar macro-dinámica
    entre dos tracks.
    """
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    block = max(1, int(sr * block_s))
    hop   = max(1, int(sr * hop_s))
    if len(mono) < block:
        return 0.0
    levels = []
    for start in range(0, len(mono) - block, hop):
        seg = mono[start:start + block]
        rms = float(np.sqrt(np.mean(seg ** 2))) + 1e-9
        db  = 20.0 * np.log10(rms)
        if db > -70.0:
            levels.append(db)
    if len(levels) < 2:
        return 0.0
    levels = np.array(levels)
    rel_gate = np.mean(levels) - 20.0
    gated = levels[levels > rel_gate]
    if len(gated) < 2:
        gated = levels
    lra = float(np.percentile(gated, 95) - np.percentile(gated, 10))
    return round(lra, 2)

# Bandas usadas para análisis/matching de dinámica y estéreo "inteligentes"
# (más anchas que las 7 bandas de `spectrum`, pensadas para separar
# graves / medios / agudos de forma robusta al filtrar).
DYNAMICS_BANDS = [("low", 20.0, 150.0), ("mid", 150.0, 2500.0), ("high", 2500.0, 20000.0)]

def _bandpass_filter(audio: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    lo = float(np.clip(lo, 1.0, sr / 2.0 - 10.0))
    hi = float(np.clip(hi, lo + 5.0, sr / 2.0 - 1.0))
    sos = butter(2, [lo, hi], btype="bandpass", fs=sr, output="sos")
    if audio.ndim == 1:
        return sosfiltfilt(sos, audio)
    return np.stack([sosfiltfilt(sos, ch) for ch in audio])

def band_crest_factors(audio: np.ndarray, sr: int, bands: list = DYNAMICS_BANDS) -> dict:
    """Crest factor (peak - RMS, en dB) por banda de frecuencia. Permite
    comparar la dinámica de graves/medios/agudos entre dos tracks por
    separado, en vez de un único crest factor de banda ancha (que puede
    esconder, por ejemplo, graves muy comprimidos con agudos muy dinámicos).
    """
    out = {}
    for name, lo, hi in bands:
        filtered = _bandpass_filter(audio, sr, lo, hi)
        mono = filtered.mean(axis=0) if filtered.ndim == 2 else filtered
        rms  = float(np.sqrt(np.mean(mono ** 2))) + 1e-9
        peak = float(np.max(np.abs(mono))) + 1e-9
        out[name] = float(20.0 * np.log10(peak) - 20.0 * np.log10(rms))
    return out

def band_stereo_correlation(audio: np.ndarray, sr: int, bands: list = DYNAMICS_BANDS) -> dict:
    """Correlación L/R por banda de frecuencia. Los masters comerciales suelen
    tener graves casi mono (correlación ~1) y agudos más anchos (correlación
    más baja); comparar banda por banda contra la referencia permite un
    matching de estéreo mucho más fiel que un único ancho global.
    """
    if audio.ndim != 2 or audio.shape[0] != 2:
        return {name: 1.0 for name, _, _ in bands}
    out = {}
    for name, lo, hi in bands:
        filtered = _bandpass_filter(audio, sr, lo, hi)
        out[name] = stereo_correlation(filtered)
    return out

def true_peak_dbfs(audio: np.ndarray, sr: int, oversample: int = 4) -> float:
    """Pico real (true peak) sobresampleado, mismo criterio que usa `limiter()`
    para detectar inter-sample peaks que el pico a sample-rate nativa no ve."""
    ovs = max(1, int(oversample))
    chans = audio if audio.ndim == 2 else audio[np.newaxis, :]
    peak = 0.0
    for ch in chans:
        up = resample_poly(ch, ovs, 1) if ovs > 1 else ch
        if len(up):
            peak = max(peak, float(np.max(np.abs(up))))
    return float(20.0 * np.log10(peak + 1e-9))


def measure_dc_offset(audio: np.ndarray) -> float:
    chans = audio if audio.ndim == 2 else audio[np.newaxis, :]
    return round(float(np.max(np.abs([np.mean(ch) for ch in chans]))), 5)


def clipping_ratio(audio: np.ndarray, threshold: float = 0.999) -> float:
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    if len(mono) == 0:
        return 0.0
    return round(float(np.mean(np.abs(mono) >= threshold)), 5)


def silence_ratio(audio: np.ndarray, sr: int, threshold_db: float = -60.0, block_s: float = 0.05) -> float:
    """Fracción del track por debajo de `threshold_db` (bloques de 50ms): útil para
    detectar silencios/fades largos que pueden estar distorsionando el LUFS integrado."""
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    block = max(1, int(sr * block_s))
    n_blocks = len(mono) // block
    if n_blocks == 0:
        return 0.0
    trimmed = mono[:n_blocks * block].reshape(n_blocks, block)
    rms = np.sqrt(np.mean(trimmed ** 2, axis=1)) + 1e-12
    db = 20.0 * np.log10(rms)
    return round(float(np.mean(db < threshold_db)), 4)


def short_term_loudness_stats(audio: np.ndarray, sr: int, block_s: float = 3.0, hop_s: float = 1.0) -> dict:
    """Estadísticas de loudness de corto plazo (ventanas de 3s, estilo momentary/
    short-term de EBU R128): máx, mín y p95. El LUFS integrado promedia todo el
    track y puede esconder picos de loudness momentáneo que sí se escuchan."""
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    block = max(1, int(sr * block_s))
    hop = max(1, int(sr * hop_s))
    if len(mono) < block:
        rms = float(np.sqrt(np.mean(mono ** 2)) + 1e-9)
        db = round(float(20.0 * np.log10(rms)), 2)
        return {"max": db, "min": db, "p95": db}
    levels = []
    for start in range(0, len(mono) - block, hop):
        seg = mono[start:start + block]
        rms = float(np.sqrt(np.mean(seg ** 2))) + 1e-9
        db = 20.0 * np.log10(rms)
        if db > -70.0:
            levels.append(db)
    if not levels:
        return {"max": -70.0, "min": -70.0, "p95": -70.0}
    arr = np.array(levels)
    return {
        "max": round(float(np.max(arr)), 2),
        "min": round(float(np.min(arr)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
    }


def spectral_shape_features(mono: np.ndarray, sr: int) -> dict:
    """Centroid, rolloff, flatness y zero-crossing rate: describen el 'brillo',
    la energía relativa en agudos y qué tan tonal (vs. ruidosa/percusiva) es la señal."""
    try:
        n_fft = min(4096, max(256, len(mono)))
        n_fft = max(64, 2 ** int(np.floor(np.log2(n_fft))))
        hop = max(1, n_fft // 4)
        centroid = librosa.feature.spectral_centroid(y=mono, sr=sr, n_fft=n_fft, hop_length=hop)
        rolloff  = librosa.feature.spectral_rolloff(y=mono, sr=sr, n_fft=n_fft, hop_length=hop, roll_percent=0.85)
        flatness = librosa.feature.spectral_flatness(y=mono, n_fft=n_fft, hop_length=hop)
        zcr      = librosa.feature.zero_crossing_rate(mono, frame_length=n_fft, hop_length=hop)
        return {
            "spectral_centroid_hz": round(float(np.mean(centroid)), 1),
            "spectral_rolloff_hz":  round(float(np.mean(rolloff)), 1),
            "spectral_flatness":    round(float(np.mean(flatness)), 4),
            "zero_crossing_rate":   round(float(np.mean(zcr)), 4),
        }
    except Exception:
        return {"spectral_centroid_hz": 0.0, "spectral_rolloff_hz": 0.0,
                "spectral_flatness": 0.0, "zero_crossing_rate": 0.0}


def transient_density(mono: np.ndarray, sr: int) -> float:
    """Onsets (ataques/transientes) por segundo: proxy de qué tan percusivo/denso
    es el material (trap/metal >> ambient/balada), usado para dosificar el transient
    shaper y el attack del compresor/limiter."""
    try:
        onsets = librosa.onset.onset_detect(y=mono, sr=sr, units="time")
        dur = max(len(mono) / sr, 1e-6)
        return round(float(len(onsets) / dur), 3)
    except Exception:
        return 0.0


def mono_compatibility_db(audio: np.ndarray) -> float:
    """Diferencia de nivel (dB) entre sumar L+R a mono y el nivel estéreo original.
    Valores muy negativos indican cancelación de fase al sumar a mono (típico de
    fase invertida o ancho estéreo artificial excesivo) — un problema serio para
    reproducción en sistemas mono (clubs, TV, bluetooth speakers)."""
    if audio.ndim != 2 or audio.shape[0] != 2:
        return 0.0
    l, r = audio[0], audio[1]
    mono_sum = (l + r) * 0.5
    stereo_rms = float(np.sqrt(np.mean(((l ** 2) + (r ** 2)) / 2.0)) + 1e-9)
    mono_rms = float(np.sqrt(np.mean(mono_sum ** 2)) + 1e-9)
    return round(float(20.0 * np.log10(mono_rms / stereo_rms)), 2)


def analyze_audio(audio: np.ndarray, sr: int) -> dict:
    """Extrae varias decenas de métricas técnicas del track: loudness (integrado,
    corto plazo, LRA), picos (sample y true peak), dinámica global y por banda,
    forma espectral (7 bandas + centroid/rolloff/flatness/ZCR), estéreo (correlación
    global y por banda, compatibilidad mono), higiene de señal (DC offset, clipping,
    silencio) y ritmo (BPM, densidad de transientes). Es el "input sensorial" que usa
    Laia (el asistente de auto-mastering) para diagnosticar el track antes de decidir
    la cadena DSP."""
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    rms  = float(np.sqrt(np.mean(mono ** 2)))
    peak = float(np.max(np.abs(mono)))
    rms_db  = float(20.0 * np.log10(rms  + 1e-9))
    peak_db = float(20.0 * np.log10(peak + 1e-9))
    lufs    = measure_lufs_integrated(audio, sr)
    dynamic_range_db = float(peak_db - rms_db)
    true_peak_db = true_peak_dbfs(audio, sr)
    try:
        tempo, _ = librosa.beat.beat_track(y=mono, sr=sr)
        bpm = float(tempo)
    except Exception:
        bpm = 0.0

    n_fft = min(4096, len(mono))
    n_fft = max(64, 2 ** int(np.floor(np.log2(n_fft))))
    avg_fft = _averaged_magnitude_spectrum(mono, n_fft)
    freqs   = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    bands = {
        "sub_bass":  (20,    80),
        "bass":      (80,    250),
        "low_mid":   (250,   500),
        "mid":       (500,   2000),
        "upper_mid": (2000,  4000),
        "presence":  (4000,  8000),
        "air":       (8000,  20000),
    }
    band_edges = [bands["sub_bass"][0]] + [hi for _, hi in bands.values()]
    band_db = _log_band_average_db(freqs, avg_fft, np.array(band_edges, dtype=float))
    spectrum = {name: round(float(db), 2) for name, db in zip(bands.keys(), band_db)}

    shape = spectral_shape_features(mono, sr)
    st_loudness = short_term_loudness_stats(audio, sr)

    return {
        # ── Loudness / nivel ────────────────────────────────────────────
        "rms_db":             round(rms_db, 2),
        "peak_db":            round(peak_db, 2),
        "true_peak_db":       round(true_peak_db, 2),
        "lufs":               round(lufs, 2),
        "plr_db":             round(true_peak_db - lufs, 2),  # peak-to-loudness ratio
        "loudness_short_term": st_loudness,
        "lra":                round(measure_lra(audio, sr), 2),
        # ── Dinámica ─────────────────────────────────────────────────────
        "dynamic_range_db":   round(dynamic_range_db, 2),
        "crest_factor_db":    round(dynamic_range_db, 2),
        "band_dynamics_db":   {k: round(v, 2) for k, v in band_crest_factors(audio, sr).items()},
        # ── Higiene de señal ────────────────────────────────────────────
        "dc_offset":          measure_dc_offset(audio),
        "clipping_ratio":     clipping_ratio(audio),
        "silence_ratio":      silence_ratio(audio, sr),
        # ── Estéreo ──────────────────────────────────────────────────────
        "stereo_correlation": round(stereo_correlation(audio), 3),
        "band_stereo_correlation": {k: round(v, 3) for k, v in band_stereo_correlation(audio, sr).items()},
        "mono_compatibility_db": mono_compatibility_db(audio),
        # ── Forma espectral / timbre ────────────────────────────────────
        "spectral_centroid_hz": shape["spectral_centroid_hz"],
        "spectral_rolloff_hz":  shape["spectral_rolloff_hz"],
        "spectral_flatness":    shape["spectral_flatness"],
        "zero_crossing_rate":   shape["zero_crossing_rate"],
        # ── Ritmo ────────────────────────────────────────────────────────
        "bpm":                round(bpm, 1),
        "transient_density":  transient_density(mono, sr),
        # ── Formato ──────────────────────────────────────────────────────
        "sample_rate":        sr,
        "channels":           1 if audio.ndim == 1 else audio.shape[0],
        "duration_sec":       round(len(mono) / sr, 2),
        # ── Espectro ─────────────────────────────────────────────────────
        "spectrum":           spectrum,
        "fft_spectrum":       spectrum_analysis_fft(audio, sr),
        # ── Resonancias / sibilancia (para Dynamic EQ / de-esser) ────────
        "resonances":         detect_resonances(audio, sr),
        "sibilance":          detect_sibilance(audio, sr),
    }

# ─── Consejos de mezcla ───────────────────────────────────────────────────────

def mix_advice(analysis: dict) -> dict:
    issues = []
    tips   = []
    score  = 100

    lufs  = analysis.get("lufs", -99)
    peak  = analysis.get("peak_db", -99)
    true_peak = analysis.get("true_peak_db", peak)
    dyn   = analysis.get("dynamic_range_db", 0)
    spec  = analysis.get("spectrum", {})
    clip_ratio = analysis.get("clipping_ratio", 0.0)
    mono_compat = analysis.get("mono_compatibility_db", 0.0)
    dc = analysis.get("dc_offset", 0.0)

    if lufs > -6:
        issues.append("Muy alto en loudness (>-6 LUFS): posible over-compression o clipping.")
        tips.append("Reducí el limiter ceiling o bajá el makeup gain del compresor.")
        score -= 20
    elif lufs > -9:
        issues.append("Loudness algo alto (-9 a -6 LUFS): al límite para streaming.")
        tips.append("Apuntá a -14 LUFS para Spotify/YouTube o -9 LUFS para club.")
        score -= 8
    elif lufs < -24:
        issues.append("Loudness muy bajo (<-24 LUFS): la mezcla se va a escuchar muy quieta.")
        tips.append("Subí el nivel general o usá normalización LUFS.")
        score -= 15
    elif lufs < -18:
        tips.append("Loudness moderadamente bajo. Podés subir con normalización LUFS a -14.")
        score -= 5

    if peak >= 0:
        issues.append("¡Clipping detectado! El pico llega a 0 dBFS.")
        tips.append("Bajá el nivel de salida o usá un limitador con ceiling más bajo.")
        score -= 25
    elif peak > -0.5:
        issues.append("Pico muy cerca de 0 dBFS (< -0.5 dB de margen).")
        tips.append("Dejá al menos 1 dB de headroom. Usá limiter ceiling = 0.95.")
        score -= 10

    if dyn < 4:
        issues.append("Rango dinámico muy comprimido (< 4 dB): mezcla 'brick-wall'.")
        tips.append("Reducí el ratio del compresor o subí el threshold.")
        score -= 18
    elif dyn < 7:
        issues.append("Rango dinámico algo limitado (4–7 dB).")
        tips.append("Probá ratio 2:1–3:1 para un resultado más natural.")
        score -= 8
    elif dyn > 25:
        tips.append("Rango dinámico muy amplio (>25 dB): podría sonar inconsistente.")
        score -= 5

    sub  = spec.get("sub_bass", -99)
    bass = spec.get("bass", -99)
    mid  = spec.get("mid", -99)
    air  = spec.get("air", -99)
    pres = spec.get("presence", -99)

    if sub > bass + 10:
        issues.append("Sub-bajos excesivos vs. bajos medios: sonido 'boomy'.")
        tips.append("Usá high-pass en 60–80 Hz y reducí EQ en 40–60 Hz.")
        score -= 12
    if bass > mid + 20:
        issues.append("Bajos dominan sobre los medios: mezcla oscura.")
        tips.append("Realzá medios (500 Hz–2 kHz) o reducí bajos en 100–200 Hz.")
        score -= 10
    if air < mid - 30:
        issues.append("Altas frecuencias muy bajas: la mezcla puede sonar apagada.")
        tips.append("Subí el high shelf (+2 a +4 dB a partir de 8 kHz).")
        score -= 8
    if pres > mid + 15:
        issues.append("Zona de presencia muy prominente (4–8 kHz): puede ser fatigante.")
        tips.append("Reducí presencia con EQ en 4–6 kHz, Q=1.5.")
        score -= 8

    if clip_ratio > 0.0005:
        issues.append(f"Clipping real detectado en {round(clip_ratio * 100, 2)}% de las muestras.")
        tips.append("Bajá el input gain o el ceiling del limiter antes de re-renderizar.")
        score -= 15
    elif true_peak > -0.3:
        issues.append(f"True peak muy cerca de 0 dBFS ({true_peak} dBTP): riesgo de inter-sample clipping en conversores D/A.")
        tips.append("Dejá al menos 1 dB de margen: usá limiter ceiling ≤ 0.94 (~-0.5 dBTP).")
        score -= 8

    if mono_compat < -3.0:
        issues.append(f"Baja compatibilidad mono ({mono_compat} dB de pérdida al sumar L+R): posible cancelación de fase.")
        tips.append("Revisá la fase del estéreo o reducí el ancho estéreo (stereo_width_amount) antes de mastering.")
        score -= 10

    if dc and abs(dc) > 0.01:
        issues.append(f"DC offset detectado ({dc}): puede reducir headroom disponible.")
        tips.append("Aplicá un filtro DC-block/high-pass muy bajo (< 20 Hz) antes de la cadena de mastering.")
        score -= 5

    if not issues:
        tips.insert(0, "¡La mezcla se ve bien técnicamente! Revisá en referencia con tracks similares.")

    score = max(0, min(100, score))
    grade = "Excelente" if score >= 85 else "Buena" if score >= 70 else "Aceptable" if score >= 50 else "Necesita trabajo"
    return {"score": score, "grade": grade, "issues": issues, "tips": tips}

# ─── Loudness targets por plataforma ──────────────────────────────────────────

PLATFORM_LOUDNESS_TARGETS = {
    "spotify":     {"lufs": -14.0, "true_peak_db": -1.0},
    "youtube":     {"lufs": -14.0, "true_peak_db": -1.0},
    "apple_music": {"lufs": -16.0, "true_peak_db": -1.0},
    "tidal":       {"lufs": -14.0, "true_peak_db": -1.0},
    "club":        {"lufs": -9.0,  "true_peak_db": -0.3},
    "cd":          {"lufs": -9.0,  "true_peak_db": -0.1},
}

def get_platform_target(platform: str) -> dict:
    return PLATFORM_LOUDNESS_TARGETS.get(platform, PLATFORM_LOUDNESS_TARGETS["spotify"])

# ─── Presets (con multibanda DESACTIVADO por defecto) ────────────────────────

MASTERING_PRESETS = {
    "rock": {
        "label": "Rock",
        "input_gain_db": 0.0, "target_peak": 0.93, "use_lufs_normalize": False, "target_lufs": -9.5,
        "hp_cutoff": 35.0, "high_shelf_gain_db": 2.0,
        "high_shelf_freq_hz": 8000.0,
        "eq1_freq": 100.0, "eq1_gain": 1.2, "eq1_q": 1.0,
        "eq2_freq": 400.0, "eq2_gain": -1.2, "eq2_q": 1.1,
        "eq3_freq": 3000.0, "eq3_gain": 1.6, "eq3_q": 1.0,
        "eq4_freq": 9000.0, "eq4_gain": 1.2, "eq4_q": 0.9,
        "comp_threshold": 0.58, "comp_ratio": 2.2, "comp_attack_ms": 12.0, "comp_release_ms": 130.0, "comp_makeup_db": 1.2,
        "transient_attack": 0.1, "transient_sustain": 0.05,
        "mb_bypass": False,
        "mb_low_crossover": 150.0, "mb_high_crossover": 4000.0,
        "mb_low_threshold": 0.6, "mb_low_ratio": 1.9, "mb_low_attack_ms": 25.0, "mb_low_release_ms": 150.0, "mb_low_makeup_db": 0.4,
        "mb_mid_threshold": 0.64, "mb_mid_ratio": 1.8, "mb_mid_attack_ms": 15.0, "mb_mid_release_ms": 120.0, "mb_mid_makeup_db": 0.4,
        "mb_high_threshold": 0.68, "mb_high_ratio": 1.6, "mb_high_attack_ms": 8.0,  "mb_high_release_ms": 90.0,  "mb_high_makeup_db": 0.3,
        "saturation_drive": 0.12, "saturation_mode": "tape", "saturation_mix": 0.25,
        "mid_gain_db": 0.0, "side_gain_db": 0.0, "stereo_width_amount": 1.04,
        "use_stereo_enhancer": False, "enhancer_bass_mono_freq": 120.0, "haas_delay_ms": 0.0,
        "reverb_size": 0.22, "reverb_wet": 0.03,
        "limiter_ceiling": 0.912, "limiter_release_ms": 70.0,
    },
    "metal": {
        "label": "Metal",
        "input_gain_db": 0.0, "target_peak": 0.94, "use_lufs_normalize": False, "target_lufs": -8.5,
        "hp_cutoff": 45.0, "high_shelf_gain_db": 2.4,
        "high_shelf_freq_hz": 7000.0,
        "eq1_freq": 90.0, "eq1_gain": 1.6, "eq1_q": 1.0,
        "eq2_freq": 300.0, "eq2_gain": -2.4, "eq2_q": 1.3,
        "eq3_freq": 2500.0, "eq3_gain": 2.4, "eq3_q": 1.1,
        "eq4_freq": 8000.0, "eq4_gain": 1.6, "eq4_q": 0.8,
        "comp_threshold": 0.52, "comp_ratio": 2.6, "comp_attack_ms": 9.0, "comp_release_ms": 110.0, "comp_makeup_db": 1.6,
        "transient_attack": 0.16, "transient_sustain": 0.0,
        "mb_bypass": False,
        "mb_low_crossover": 180.0, "mb_high_crossover": 3500.0,
        "mb_low_threshold": 0.55, "mb_low_ratio": 2.2, "mb_low_attack_ms": 20.0, "mb_low_release_ms": 140.0, "mb_low_makeup_db": 0.4,
        "mb_mid_threshold": 0.58, "mb_mid_ratio": 2.0, "mb_mid_attack_ms": 12.0, "mb_mid_release_ms": 100.0, "mb_mid_makeup_db": 0.6,
        "mb_high_threshold": 0.63, "mb_high_ratio": 1.8, "mb_high_attack_ms": 6.0,  "mb_high_release_ms": 70.0,  "mb_high_makeup_db": 0.4,
        "saturation_drive": 0.2, "saturation_mode": "tube", "saturation_mix": 0.28,
        "mid_gain_db": 0.0, "side_gain_db": 0.0, "stereo_width_amount": 1.0,
        "use_stereo_enhancer": False, "enhancer_bass_mono_freq": 120.0, "haas_delay_ms": 0.0,
        "reverb_size": 0.18, "reverb_wet": 0.02,
        "limiter_ceiling": 0.912, "limiter_release_ms": 55.0,
    },
    "trap": {
        "label": "Trap",
        "input_gain_db": 0.0, "target_peak": 0.94, "use_lufs_normalize": False, "target_lufs": -7.5,
        "hp_cutoff": 25.0, "high_shelf_gain_db": 2.0,
        "high_shelf_freq_hz": 10000.0,
        "eq1_freq": 60.0, "eq1_gain": 2.4, "eq1_q": 0.9,
        "eq2_freq": 250.0, "eq2_gain": -1.6, "eq2_q": 1.2,
        "eq3_freq": 3500.0, "eq3_gain": 2.0, "eq3_q": 1.0,
        "eq4_freq": 10000.0, "eq4_gain": 2.0, "eq4_q": 0.8,
        "comp_threshold": 0.46, "comp_ratio": 2.8, "comp_attack_ms": 6.0, "comp_release_ms": 100.0, "comp_makeup_db": 1.6,
        "transient_attack": 0.2, "transient_sustain": -0.08,
        "mb_bypass": False,
        "mb_low_crossover": 100.0, "mb_high_crossover": 4500.0,
        "mb_low_threshold": 0.48, "mb_low_ratio": 2.6, "mb_low_attack_ms": 15.0, "mb_low_release_ms": 170.0, "mb_low_makeup_db": 0.8,
        "mb_mid_threshold": 0.62, "mb_mid_ratio": 1.9, "mb_mid_attack_ms": 10.0, "mb_mid_release_ms": 100.0, "mb_mid_makeup_db": 0.4,
        "mb_high_threshold": 0.67, "mb_high_ratio": 1.7, "mb_high_attack_ms": 5.0,  "mb_high_release_ms": 60.0,  "mb_high_makeup_db": 0.3,
        "saturation_drive": 0.1, "saturation_mode": "tape", "saturation_mix": 0.18,
        "mid_gain_db": 0.0, "side_gain_db": 0.0, "stereo_width_amount": 1.08,
        "use_stereo_enhancer": True, "enhancer_bass_mono_freq": 100.0, "haas_delay_ms": 3.0,
        "reverb_size": 0.18, "reverb_wet": 0.0,
        "limiter_ceiling": 0.912, "limiter_release_ms": 45.0,
    },
    "rap": {
        "label": "Rap / Hip-Hop",
        "input_gain_db": 0.0, "target_peak": 0.93, "use_lufs_normalize": False, "target_lufs": -8.5,
        "hp_cutoff": 32.0, "high_shelf_gain_db": 1.4,
        "high_shelf_freq_hz": 8000.0,
        "eq1_freq": 80.0, "eq1_gain": 2.0, "eq1_q": 0.9,
        "eq2_freq": 350.0, "eq2_gain": -1.2, "eq2_q": 1.1,
        "eq3_freq": 2000.0, "eq3_gain": 1.6, "eq3_q": 1.2,
        "eq4_freq": 7000.0, "eq4_gain": 1.2, "eq4_q": 0.9,
        "comp_threshold": 0.56, "comp_ratio": 2.2, "comp_attack_ms": 11.0, "comp_release_ms": 115.0, "comp_makeup_db": 1.2,
        "transient_attack": 0.13, "transient_sustain": 0.0,
        "mb_bypass": False,
        "mb_low_crossover": 120.0, "mb_high_crossover": 4000.0,
        "mb_low_threshold": 0.54, "mb_low_ratio": 2.2, "mb_low_attack_ms": 18.0, "mb_low_release_ms": 160.0, "mb_low_makeup_db": 0.5,
        "mb_mid_threshold": 0.62, "mb_mid_ratio": 1.8, "mb_mid_attack_ms": 12.0, "mb_mid_release_ms": 110.0, "mb_mid_makeup_db": 0.4,
        "mb_high_threshold": 0.67, "mb_high_ratio": 1.6, "mb_high_attack_ms": 6.0,  "mb_high_release_ms": 80.0,  "mb_high_makeup_db": 0.3,
        "saturation_drive": 0.1, "saturation_mode": "tape", "saturation_mix": 0.2,
        "mid_gain_db": 0.0, "side_gain_db": 0.0, "stereo_width_amount": 1.0,
        "use_stereo_enhancer": False, "enhancer_bass_mono_freq": 110.0, "haas_delay_ms": 0.0,
        "reverb_size": 0.18, "reverb_wet": 0.03,
        "limiter_ceiling": 0.912, "limiter_release_ms": 65.0,
    },
    "reggaeton": {
        "label": "Reggaeton",
        "input_gain_db": 0.0, "target_peak": 0.94, "use_lufs_normalize": False, "target_lufs": -8.0,
        "hp_cutoff": 28.0, "high_shelf_gain_db": 2.0,
        "high_shelf_freq_hz": 9000.0,
        "eq1_freq": 70.0, "eq1_gain": 2.4, "eq1_q": 0.9,
        "eq2_freq": 300.0, "eq2_gain": -1.6, "eq2_q": 1.2,
        "eq3_freq": 3000.0, "eq3_gain": 2.0, "eq3_q": 1.0,
        "eq4_freq": 9000.0, "eq4_gain": 1.6, "eq4_q": 0.8,
        "comp_threshold": 0.5, "comp_ratio": 2.5, "comp_attack_ms": 9.0, "comp_release_ms": 105.0, "comp_makeup_db": 1.4,
        "transient_attack": 0.17, "transient_sustain": -0.04,
        "mb_bypass": False,
        "mb_low_crossover": 110.0, "mb_high_crossover": 4200.0,
        "mb_low_threshold": 0.52, "mb_low_ratio": 2.4, "mb_low_attack_ms": 16.0, "mb_low_release_ms": 170.0, "mb_low_makeup_db": 0.6,
        "mb_mid_threshold": 0.6, "mb_mid_ratio": 1.9, "mb_mid_attack_ms": 10.0, "mb_mid_release_ms": 110.0, "mb_mid_makeup_db": 0.4,
        "mb_high_threshold": 0.66, "mb_high_ratio": 1.7, "mb_high_attack_ms": 6.0,  "mb_high_release_ms": 75.0,  "mb_high_makeup_db": 0.3,
        "saturation_drive": 0.1, "saturation_mode": "tape", "saturation_mix": 0.2,
        "mid_gain_db": 0.0, "side_gain_db": 0.0, "stereo_width_amount": 1.12,
        "use_stereo_enhancer": True, "enhancer_bass_mono_freq": 110.0, "haas_delay_ms": 3.0,
        "reverb_size": 0.15, "reverb_wet": 0.02,
        "limiter_ceiling": 0.912, "limiter_release_ms": 50.0,
    },
    "pop": {
        "label": "Pop",
        "input_gain_db": 0.0, "target_peak": 0.92, "use_lufs_normalize": False, "target_lufs": -10.0,
        "hp_cutoff": 35.0, "high_shelf_gain_db": 2.0,
        "high_shelf_freq_hz": 10000.0,
        "eq1_freq": 100.0, "eq1_gain": 0.8, "eq1_q": 1.0,
        "eq2_freq": 400.0, "eq2_gain": -0.8, "eq2_q": 1.1,
        "eq3_freq": 3000.0, "eq3_gain": 1.6, "eq3_q": 1.0,
        "eq4_freq": 10000.0, "eq4_gain": 2.0, "eq4_q": 0.8,
        "comp_threshold": 0.6, "comp_ratio": 2.0, "comp_attack_ms": 13.0, "comp_release_ms": 135.0, "comp_makeup_db": 1.0,
        "transient_attack": 0.08, "transient_sustain": 0.0,
        "mb_bypass": False,
        "mb_low_crossover": 200.0, "mb_high_crossover": 4000.0,
        "mb_low_threshold": 0.64, "mb_low_ratio": 1.8, "mb_low_attack_ms": 22.0, "mb_low_release_ms": 150.0, "mb_low_makeup_db": 0.3,
        "mb_mid_threshold": 0.68, "mb_mid_ratio": 1.6, "mb_mid_attack_ms": 14.0, "mb_mid_release_ms": 120.0, "mb_mid_makeup_db": 0.3,
        "mb_high_threshold": 0.7, "mb_high_ratio": 1.5, "mb_high_attack_ms": 8.0,  "mb_high_release_ms": 90.0,  "mb_high_makeup_db": 0.2,
        "saturation_drive": 0.07, "saturation_mode": "tape", "saturation_mix": 0.16,
        "mid_gain_db": 0.0, "side_gain_db": 0.0, "stereo_width_amount": 1.08,
        "use_stereo_enhancer": True, "enhancer_bass_mono_freq": 120.0, "haas_delay_ms": 3.0,
        "reverb_size": 0.28, "reverb_wet": 0.05,
        "limiter_ceiling": 0.912, "limiter_release_ms": 75.0,
    },
    "cd": {
        "label": "CD / Audiófilo",
        "input_gain_db": 0.0, "target_peak": 0.87, "use_lufs_normalize": False, "target_lufs": -16.0,
        "hp_cutoff": 22.0, "high_shelf_gain_db": 0.6,
        "high_shelf_freq_hz": 8000.0,
        "eq1_freq": 100.0, "eq1_gain": 0.0, "eq1_q": 1.0,
        "eq2_freq": 500.0, "eq2_gain": 0.0, "eq2_q": 1.0,
        "eq3_freq": 2000.0, "eq3_gain": 0.3, "eq3_q": 1.0,
        "eq4_freq": 9000.0, "eq4_gain": 0.3, "eq4_q": 0.9,
        "comp_threshold": 0.82, "comp_ratio": 1.4, "comp_attack_ms": 22.0, "comp_release_ms": 190.0, "comp_makeup_db": 0.0,
        "transient_attack": 0.0, "transient_sustain": 0.0,
        # Multibanda apagado a propósito: filosofía "audiófila" = mínima compresión, máxima dinámica.
        "mb_bypass": True,
        "mb_low_crossover": 250.0, "mb_high_crossover": 4000.0,
        "mb_low_threshold": 0.85, "mb_low_ratio": 1.4, "mb_low_attack_ms": 30.0, "mb_low_release_ms": 200.0, "mb_low_makeup_db": 0.0,
        "mb_mid_threshold": 0.85, "mb_mid_ratio": 1.4, "mb_mid_attack_ms": 20.0, "mb_mid_release_ms": 180.0, "mb_mid_makeup_db": 0.0,
        "mb_high_threshold": 0.85, "mb_high_ratio": 1.3, "mb_high_attack_ms": 10.0,  "mb_high_release_ms": 120.0,  "mb_high_makeup_db": 0.0,
        "saturation_drive": 0.0, "saturation_mode": "tape", "saturation_mix": 0.0,
        "mid_gain_db": 0.0, "side_gain_db": 0.0, "stereo_width_amount": 1.0,
        "use_stereo_enhancer": False, "enhancer_bass_mono_freq": 120.0, "haas_delay_ms": 0.0,
        "reverb_size": 0.28, "reverb_wet": 0.02,
        "limiter_ceiling": 0.891, "limiter_release_ms": 110.0,
    },
    "edm": {
        "label": "EDM / House",
        "input_gain_db": 0.0, "target_peak": 0.95, "use_lufs_normalize": False, "target_lufs": -7.0,
        "hp_cutoff": 28.0, "high_shelf_gain_db": 2.4,
        "high_shelf_freq_hz": 11000.0,
        "eq1_freq": 60.0, "eq1_gain": 2.0, "eq1_q": 0.9,
        "eq2_freq": 250.0, "eq2_gain": -1.2, "eq2_q": 1.2,
        "eq3_freq": 3500.0, "eq3_gain": 1.8, "eq3_q": 1.0,
        "eq4_freq": 11000.0, "eq4_gain": 2.4, "eq4_q": 0.8,
        "comp_threshold": 0.46, "comp_ratio": 2.8, "comp_attack_ms": 7.0, "comp_release_ms": 95.0, "comp_makeup_db": 1.6,
        "transient_attack": 0.17, "transient_sustain": -0.08,
        "mb_bypass": False,
        "mb_low_crossover": 100.0, "mb_high_crossover": 5000.0,
        "mb_low_threshold": 0.48, "mb_low_ratio": 2.6, "mb_low_attack_ms": 12.0, "mb_low_release_ms": 180.0, "mb_low_makeup_db": 0.8,
        "mb_mid_threshold": 0.62, "mb_mid_ratio": 1.9, "mb_mid_attack_ms": 8.0, "mb_mid_release_ms": 90.0, "mb_mid_makeup_db": 0.4,
        "mb_high_threshold": 0.67, "mb_high_ratio": 1.7, "mb_high_attack_ms": 5.0,  "mb_high_release_ms": 60.0,  "mb_high_makeup_db": 0.3,
        "saturation_drive": 0.09, "saturation_mode": "tube", "saturation_mix": 0.18,
        "mid_gain_db": 0.0, "side_gain_db": 0.8, "stereo_width_amount": 1.16,
        "use_stereo_enhancer": True, "enhancer_bass_mono_freq": 100.0, "haas_delay_ms": 5.0,
        "reverb_size": 0.15, "reverb_wet": 0.02,
        "limiter_ceiling": 0.933, "limiter_release_ms": 40.0,
    },
    "techno": {
        "label": "Techno",
        "input_gain_db": 0.0, "target_peak": 0.95, "use_lufs_normalize": False, "target_lufs": -7.0,
        "hp_cutoff": 32.0, "high_shelf_gain_db": 1.8,
        "high_shelf_freq_hz": 9000.0,
        "eq1_freq": 55.0, "eq1_gain": 1.6, "eq1_q": 0.9,
        "eq2_freq": 220.0, "eq2_gain": -1.6, "eq2_q": 1.3,
        "eq3_freq": 3000.0, "eq3_gain": 1.2, "eq3_q": 1.0,
        "eq4_freq": 9000.0, "eq4_gain": 0.8, "eq4_q": 0.8,
        "comp_threshold": 0.44, "comp_ratio": 2.8, "comp_attack_ms": 7.0, "comp_release_ms": 95.0, "comp_makeup_db": 1.6,
        "transient_attack": 0.13, "transient_sustain": -0.12,
        "mb_bypass": False,
        "mb_low_crossover": 90.0, "mb_high_crossover": 4500.0,
        "mb_low_threshold": 0.46, "mb_low_ratio": 2.8, "mb_low_attack_ms": 10.0, "mb_low_release_ms": 170.0, "mb_low_makeup_db": 0.9,
        "mb_mid_threshold": 0.6, "mb_mid_ratio": 1.9, "mb_mid_attack_ms": 8.0, "mb_mid_release_ms": 90.0, "mb_mid_makeup_db": 0.4,
        "mb_high_threshold": 0.65, "mb_high_ratio": 1.7, "mb_high_attack_ms": 5.0,  "mb_high_release_ms": 60.0,  "mb_high_makeup_db": 0.3,
        "saturation_drive": 0.13, "saturation_mode": "tube", "saturation_mix": 0.22,
        "mid_gain_db": 0.0, "side_gain_db": 0.0, "stereo_width_amount": 1.05,
        "use_stereo_enhancer": True, "enhancer_bass_mono_freq": 90.0, "haas_delay_ms": 3.0,
        "reverb_size": 0.15, "reverb_wet": 0.0,
        "limiter_ceiling": 0.933, "limiter_release_ms": 40.0,
    },
    "rnb": {
        "label": "R&B / Soul",
        "input_gain_db": 0.0, "target_peak": 0.91, "use_lufs_normalize": False, "target_lufs": -10.5,
        "hp_cutoff": 30.0, "high_shelf_gain_db": 1.4,
        "high_shelf_freq_hz": 8000.0,
        "eq1_freq": 90.0, "eq1_gain": 1.2, "eq1_q": 0.9,
        "eq2_freq": 350.0, "eq2_gain": -1.0, "eq2_q": 1.1,
        "eq3_freq": 2500.0, "eq3_gain": 1.2, "eq3_q": 1.0,
        "eq4_freq": 8000.0, "eq4_gain": 1.4, "eq4_q": 0.9,
        "comp_threshold": 0.6, "comp_ratio": 1.9, "comp_attack_ms": 15.0, "comp_release_ms": 145.0, "comp_makeup_db": 0.9,
        "transient_attack": 0.06, "transient_sustain": 0.05,
        "mb_bypass": False,
        "mb_low_crossover": 150.0, "mb_high_crossover": 3800.0,
        "mb_low_threshold": 0.6, "mb_low_ratio": 1.9, "mb_low_attack_ms": 22.0, "mb_low_release_ms": 160.0, "mb_low_makeup_db": 0.4,
        "mb_mid_threshold": 0.64, "mb_mid_ratio": 1.7, "mb_mid_attack_ms": 16.0, "mb_mid_release_ms": 130.0, "mb_mid_makeup_db": 0.4,
        "mb_high_threshold": 0.71, "mb_high_ratio": 1.5, "mb_high_attack_ms": 9.0,  "mb_high_release_ms": 90.0,  "mb_high_makeup_db": 0.2,
        "saturation_drive": 0.14, "saturation_mode": "tape", "saturation_mix": 0.28,
        "mid_gain_db": 0.0, "side_gain_db": 0.4, "stereo_width_amount": 1.05,
        "use_stereo_enhancer": True, "enhancer_bass_mono_freq": 110.0, "haas_delay_ms": 3.0,
        "reverb_size": 0.28, "reverb_wet": 0.05,
        "limiter_ceiling": 0.891, "limiter_release_ms": 100.0,
    },
    "jazz": {
        "label": "Jazz",
        "input_gain_db": 0.0, "target_peak": 0.88, "use_lufs_normalize": False, "target_lufs": -15.5,
        "hp_cutoff": 22.0, "high_shelf_gain_db": 1.0,
        "high_shelf_freq_hz": 8000.0,
        "eq1_freq": 100.0, "eq1_gain": 0.4, "eq1_q": 1.0,
        "eq2_freq": 450.0, "eq2_gain": -0.4, "eq2_q": 1.0,
        "eq3_freq": 2500.0, "eq3_gain": 0.6, "eq3_q": 1.0,
        "eq4_freq": 8000.0, "eq4_gain": 0.8, "eq4_q": 0.9,
        "comp_threshold": 0.78, "comp_ratio": 1.4, "comp_attack_ms": 22.0, "comp_release_ms": 190.0, "comp_makeup_db": 0.3,
        "transient_attack": 0.0, "transient_sustain": 0.05,
        "mb_bypass": False,
        "mb_low_crossover": 180.0, "mb_high_crossover": 4000.0,
        "mb_low_threshold": 0.82, "mb_low_ratio": 1.3, "mb_low_attack_ms": 30.0, "mb_low_release_ms": 200.0, "mb_low_makeup_db": 0.0,
        "mb_mid_threshold": 0.84, "mb_mid_ratio": 1.3, "mb_mid_attack_ms": 22.0, "mb_mid_release_ms": 180.0, "mb_mid_makeup_db": 0.0,
        "mb_high_threshold": 0.86, "mb_high_ratio": 1.25, "mb_high_attack_ms": 12.0,  "mb_high_release_ms": 130.0,  "mb_high_makeup_db": 0.0,
        "saturation_drive": 0.06, "saturation_mode": "tape", "saturation_mix": 0.15,
        "mid_gain_db": 0.0, "side_gain_db": 0.0, "stereo_width_amount": 1.0,
        "use_stereo_enhancer": False, "enhancer_bass_mono_freq": 120.0, "haas_delay_ms": 0.0,
        "reverb_size": 0.38, "reverb_wet": 0.06,
        "limiter_ceiling": 0.867, "limiter_release_ms": 160.0,
    },
    "classical": {
        "label": "Clásica / Orquesta",
        "input_gain_db": 0.0, "target_peak": 0.85, "use_lufs_normalize": False, "target_lufs": -18.0,
        "hp_cutoff": 18.0, "high_shelf_gain_db": 0.3,
        "high_shelf_freq_hz": 9000.0,
        "eq1_freq": 100.0, "eq1_gain": 0.0, "eq1_q": 1.0,
        "eq2_freq": 500.0, "eq2_gain": 0.0, "eq2_q": 1.0,
        "eq3_freq": 2500.0, "eq3_gain": 0.2, "eq3_q": 1.0,
        "eq4_freq": 9000.0, "eq4_gain": 0.2, "eq4_q": 0.9,
        "comp_threshold": 0.88, "comp_ratio": 1.25, "comp_attack_ms": 28.0, "comp_release_ms": 220.0, "comp_makeup_db": 0.0,
        "transient_attack": 0.0, "transient_sustain": 0.0,
        # Multibanda apagado a propósito: la música orquestal se masteriza casi sin compresión.
        "mb_bypass": True,
        "mb_low_crossover": 150.0, "mb_high_crossover": 5000.0,
        "mb_low_threshold": 0.9, "mb_low_ratio": 1.2, "mb_low_attack_ms": 40.0, "mb_low_release_ms": 250.0, "mb_low_makeup_db": 0.0,
        "mb_mid_threshold": 0.9, "mb_mid_ratio": 1.2, "mb_mid_attack_ms": 30.0, "mb_mid_release_ms": 220.0, "mb_mid_makeup_db": 0.0,
        "mb_high_threshold": 0.91, "mb_high_ratio": 1.15, "mb_high_attack_ms": 15.0,  "mb_high_release_ms": 150.0,  "mb_high_makeup_db": 0.0,
        "saturation_drive": 0.0, "saturation_mode": "tape", "saturation_mix": 0.0,
        "mid_gain_db": 0.0, "side_gain_db": 0.0, "stereo_width_amount": 1.0,
        "use_stereo_enhancer": False, "enhancer_bass_mono_freq": 120.0, "haas_delay_ms": 0.0,
        "reverb_size": 0.45, "reverb_wet": 0.08,
        "limiter_ceiling": 0.851, "limiter_release_ms": 260.0,
    },
    "lofi": {
        "label": "Lo-Fi",
        "input_gain_db": 0.0, "target_peak": 0.9, "use_lufs_normalize": False, "target_lufs": -11.5,
        "hp_cutoff": 40.0, "high_shelf_gain_db": -2.4,
        "high_shelf_freq_hz": 6000.0,
        "eq1_freq": 80.0, "eq1_gain": -0.8, "eq1_q": 0.9,
        "eq2_freq": 300.0, "eq2_gain": 0.8, "eq2_q": 1.1,
        "eq3_freq": 2000.0, "eq3_gain": -1.2, "eq3_q": 1.0,
        "eq4_freq": 6000.0, "eq4_gain": -2.4, "eq4_q": 0.8,
        "comp_threshold": 0.6, "comp_ratio": 1.8, "comp_attack_ms": 18.0, "comp_release_ms": 150.0, "comp_makeup_db": 0.8,
        "transient_attack": 0.04, "transient_sustain": 0.16,
        "mb_bypass": False,
        "mb_low_crossover": 140.0, "mb_high_crossover": 3500.0,
        "mb_low_threshold": 0.64, "mb_low_ratio": 1.8, "mb_low_attack_ms": 25.0, "mb_low_release_ms": 170.0, "mb_low_makeup_db": 0.2,
        "mb_mid_threshold": 0.6, "mb_mid_ratio": 1.9, "mb_mid_attack_ms": 18.0, "mb_mid_release_ms": 140.0, "mb_mid_makeup_db": 0.4,
        "mb_high_threshold": 0.73, "mb_high_ratio": 1.4, "mb_high_attack_ms": 10.0,  "mb_high_release_ms": 100.0,  "mb_high_makeup_db": 0.0,
        "saturation_drive": 0.28, "saturation_mode": "tape", "saturation_mix": 0.4,
        "mid_gain_db": 0.0, "side_gain_db": -0.8, "stereo_width_amount": 0.92,
        "use_stereo_enhancer": False, "enhancer_bass_mono_freq": 130.0, "haas_delay_ms": 0.0,
        "reverb_size": 0.32, "reverb_wet": 0.08,
        "limiter_ceiling": 0.891, "limiter_release_ms": 130.0,
    },
    "acoustic": {
        "label": "Acústico / Folk",
        "input_gain_db": 0.0, "target_peak": 0.88, "use_lufs_normalize": False, "target_lufs": -13.0,
        "hp_cutoff": 28.0, "high_shelf_gain_db": 1.2,
        "high_shelf_freq_hz": 9000.0,
        "eq1_freq": 110.0, "eq1_gain": 0.6, "eq1_q": 1.0,
        "eq2_freq": 380.0, "eq2_gain": -0.8, "eq2_q": 1.1,
        "eq3_freq": 2800.0, "eq3_gain": 1.0, "eq3_q": 1.0,
        "eq4_freq": 9000.0, "eq4_gain": 1.2, "eq4_q": 0.9,
        "comp_threshold": 0.68, "comp_ratio": 1.7, "comp_attack_ms": 18.0, "comp_release_ms": 160.0, "comp_makeup_db": 0.6,
        "transient_attack": 0.04, "transient_sustain": 0.08,
        "mb_bypass": False,
        "mb_low_crossover": 160.0, "mb_high_crossover": 4000.0,
        "mb_low_threshold": 0.72, "mb_low_ratio": 1.6, "mb_low_attack_ms": 28.0, "mb_low_release_ms": 190.0, "mb_low_makeup_db": 0.2,
        "mb_mid_threshold": 0.74, "mb_mid_ratio": 1.5, "mb_mid_attack_ms": 18.0, "mb_mid_release_ms": 150.0, "mb_mid_makeup_db": 0.2,
        "mb_high_threshold": 0.78, "mb_high_ratio": 1.4, "mb_high_attack_ms": 10.0,  "mb_high_release_ms": 110.0,  "mb_high_makeup_db": 0.1,
        "saturation_drive": 0.08, "saturation_mode": "tape", "saturation_mix": 0.18,
        "mid_gain_db": 0.0, "side_gain_db": 0.3, "stereo_width_amount": 1.05,
        "use_stereo_enhancer": False, "enhancer_bass_mono_freq": 120.0, "haas_delay_ms": 0.0,
        "reverb_size": 0.32, "reverb_wet": 0.07,
        "limiter_ceiling": 0.891, "limiter_release_ms": 140.0,
    },
    "indie": {
        "label": "Indie / Alternativo",
        "input_gain_db": 0.0, "target_peak": 0.91, "use_lufs_normalize": False, "target_lufs": -10.5,
        "hp_cutoff": 32.0, "high_shelf_gain_db": 1.8,
        "high_shelf_freq_hz": 9500.0,
        "eq1_freq": 100.0, "eq1_gain": 0.8, "eq1_q": 1.0,
        "eq2_freq": 420.0, "eq2_gain": -1.0, "eq2_q": 1.1,
        "eq3_freq": 2800.0, "eq3_gain": 1.4, "eq3_q": 1.0,
        "eq4_freq": 9500.0, "eq4_gain": 1.4, "eq4_q": 0.8,
        "comp_threshold": 0.6, "comp_ratio": 2.0, "comp_attack_ms": 14.0, "comp_release_ms": 135.0, "comp_makeup_db": 0.9,
        "transient_attack": 0.09, "transient_sustain": 0.04,
        "mb_bypass": False,
        "mb_low_crossover": 160.0, "mb_high_crossover": 4000.0,
        "mb_low_threshold": 0.64, "mb_low_ratio": 1.8, "mb_low_attack_ms": 24.0, "mb_low_release_ms": 155.0, "mb_low_makeup_db": 0.3,
        "mb_mid_threshold": 0.66, "mb_mid_ratio": 1.7, "mb_mid_attack_ms": 15.0, "mb_mid_release_ms": 125.0, "mb_mid_makeup_db": 0.3,
        "mb_high_threshold": 0.7, "mb_high_ratio": 1.5, "mb_high_attack_ms": 8.0,  "mb_high_release_ms": 90.0,  "mb_high_makeup_db": 0.2,
        "saturation_drive": 0.14, "saturation_mode": "tape", "saturation_mix": 0.26,
        "mid_gain_db": 0.0, "side_gain_db": 0.2, "stereo_width_amount": 1.06,
        "use_stereo_enhancer": False, "enhancer_bass_mono_freq": 115.0, "haas_delay_ms": 0.0,
        "reverb_size": 0.26, "reverb_wet": 0.04,
        "limiter_ceiling": 0.905, "limiter_release_ms": 85.0,
    },
    "cumbia": {
        "label": "Cumbia",
        "input_gain_db": 0.0, "target_peak": 0.93, "use_lufs_normalize": False, "target_lufs": -8.5,
        "hp_cutoff": 30.0, "high_shelf_gain_db": 2.0,
        "high_shelf_freq_hz": 8500.0,
        "eq1_freq": 75.0, "eq1_gain": 2.0, "eq1_q": 0.9,
        "eq2_freq": 320.0, "eq2_gain": -1.4, "eq2_q": 1.2,
        "eq3_freq": 2800.0, "eq3_gain": 1.8, "eq3_q": 1.0,
        "eq4_freq": 8500.0, "eq4_gain": 1.6, "eq4_q": 0.8,
        "comp_threshold": 0.52, "comp_ratio": 2.4, "comp_attack_ms": 10.0, "comp_release_ms": 110.0, "comp_makeup_db": 1.3,
        "transient_attack": 0.15, "transient_sustain": -0.02,
        "mb_bypass": False,
        "mb_low_crossover": 115.0, "mb_high_crossover": 4200.0,
        "mb_low_threshold": 0.54, "mb_low_ratio": 2.2, "mb_low_attack_ms": 17.0, "mb_low_release_ms": 165.0, "mb_low_makeup_db": 0.5,
        "mb_mid_threshold": 0.6, "mb_mid_ratio": 1.9, "mb_mid_attack_ms": 11.0, "mb_mid_release_ms": 115.0, "mb_mid_makeup_db": 0.4,
        "mb_high_threshold": 0.66, "mb_high_ratio": 1.7, "mb_high_attack_ms": 6.0,  "mb_high_release_ms": 80.0,  "mb_high_makeup_db": 0.3,
        "saturation_drive": 0.12, "saturation_mode": "tape", "saturation_mix": 0.22,
        "mid_gain_db": 0.0, "side_gain_db": 0.3, "stereo_width_amount": 1.1,
        "use_stereo_enhancer": True, "enhancer_bass_mono_freq": 110.0, "haas_delay_ms": 2.0,
        "reverb_size": 0.2, "reverb_wet": 0.04,
        "limiter_ceiling": 0.912, "limiter_release_ms": 55.0,
    },
    "funk": {
        "label": "Funk / Disco",
        "input_gain_db": 0.0, "target_peak": 0.92, "use_lufs_normalize": False, "target_lufs": -9.0,
        "hp_cutoff": 32.0, "high_shelf_gain_db": 1.8,
        "high_shelf_freq_hz": 9000.0,
        "eq1_freq": 85.0, "eq1_gain": 1.6, "eq1_q": 0.9,
        "eq2_freq": 350.0, "eq2_gain": -1.2, "eq2_q": 1.2,
        "eq3_freq": 2600.0, "eq3_gain": 1.6, "eq3_q": 1.0,
        "eq4_freq": 9000.0, "eq4_gain": 1.6, "eq4_q": 0.8,
        "comp_threshold": 0.54, "comp_ratio": 2.3, "comp_attack_ms": 8.0, "comp_release_ms": 95.0, "comp_makeup_db": 1.2,
        "transient_attack": 0.14, "transient_sustain": 0.02,
        "mb_bypass": False,
        "mb_low_crossover": 130.0, "mb_high_crossover": 4200.0,
        "mb_low_threshold": 0.56, "mb_low_ratio": 2.1, "mb_low_attack_ms": 14.0, "mb_low_release_ms": 140.0, "mb_low_makeup_db": 0.4,
        "mb_mid_threshold": 0.6, "mb_mid_ratio": 1.9, "mb_mid_attack_ms": 9.0, "mb_mid_release_ms": 100.0, "mb_mid_makeup_db": 0.4,
        "mb_high_threshold": 0.66, "mb_high_ratio": 1.7, "mb_high_attack_ms": 5.0,  "mb_high_release_ms": 70.0,  "mb_high_makeup_db": 0.3,
        "saturation_drive": 0.16, "saturation_mode": "tube", "saturation_mix": 0.3,
        "mid_gain_db": 0.0, "side_gain_db": 0.4, "stereo_width_amount": 1.1,
        "use_stereo_enhancer": True, "enhancer_bass_mono_freq": 105.0, "haas_delay_ms": 2.0,
        "reverb_size": 0.2, "reverb_wet": 0.04,
        "limiter_ceiling": 0.912, "limiter_release_ms": 60.0,
    },
    "ambient": {
        "label": "Ambient / Chill",
        "input_gain_db": 0.0, "target_peak": 0.85, "use_lufs_normalize": False, "target_lufs": -15.0,
        "hp_cutoff": 20.0, "high_shelf_gain_db": 0.8,
        "high_shelf_freq_hz": 10000.0,
        "eq1_freq": 90.0, "eq1_gain": 0.4, "eq1_q": 1.0,
        "eq2_freq": 400.0, "eq2_gain": -0.6, "eq2_q": 1.0,
        "eq3_freq": 2500.0, "eq3_gain": 0.6, "eq3_q": 1.0,
        "eq4_freq": 10000.0, "eq4_gain": 1.0, "eq4_q": 0.8,
        "comp_threshold": 0.76, "comp_ratio": 1.4, "comp_attack_ms": 25.0, "comp_release_ms": 210.0, "comp_makeup_db": 0.3,
        "transient_attack": 0.0, "transient_sustain": 0.1,
        "mb_bypass": False,
        "mb_low_crossover": 170.0, "mb_high_crossover": 4200.0,
        "mb_low_threshold": 0.8, "mb_low_ratio": 1.3, "mb_low_attack_ms": 32.0, "mb_low_release_ms": 220.0, "mb_low_makeup_db": 0.0,
        "mb_mid_threshold": 0.82, "mb_mid_ratio": 1.3, "mb_mid_attack_ms": 24.0, "mb_mid_release_ms": 190.0, "mb_mid_makeup_db": 0.0,
        "mb_high_threshold": 0.85, "mb_high_ratio": 1.25, "mb_high_attack_ms": 14.0,  "mb_high_release_ms": 140.0,  "mb_high_makeup_db": 0.0,
        "saturation_drive": 0.05, "saturation_mode": "tape", "saturation_mix": 0.12,
        "mid_gain_db": 0.0, "side_gain_db": 0.6, "stereo_width_amount": 1.15,
        "use_stereo_enhancer": True, "enhancer_bass_mono_freq": 100.0, "haas_delay_ms": 6.0,
        "reverb_size": 0.55, "reverb_wet": 0.14,
        "limiter_ceiling": 0.867, "limiter_release_ms": 180.0,
    },
    "podcast": {
        "label": "Podcast / Voz",
        "input_gain_db": 0.0, "target_peak": 0.89, "use_lufs_normalize": True, "target_lufs": -16.0,
        "hp_cutoff": 80.0, "high_shelf_gain_db": 1.0,
        "high_shelf_freq_hz": 8000.0,
        "eq1_freq": 120.0, "eq1_gain": -1.0, "eq1_q": 1.0,
        "eq2_freq": 350.0, "eq2_gain": -1.5, "eq2_q": 1.2,
        "eq3_freq": 3200.0, "eq3_gain": 2.0, "eq3_q": 1.1,
        "eq4_freq": 8000.0, "eq4_gain": 1.0, "eq4_q": 0.9,
        "comp_threshold": 0.55, "comp_ratio": 2.6, "comp_attack_ms": 8.0, "comp_release_ms": 120.0, "comp_makeup_db": 1.5,
        "transient_attack": 0.0, "transient_sustain": 0.0,
        # Multibanda apagado: la voz hablada se beneficia de compresión simple, no multibanda.
        "mb_bypass": True,
        "mb_low_crossover": 200.0, "mb_high_crossover": 4000.0,
        "mb_low_threshold": 0.7, "mb_low_ratio": 1.8, "mb_low_attack_ms": 20.0, "mb_low_release_ms": 150.0, "mb_low_makeup_db": 0.0,
        "mb_mid_threshold": 0.7, "mb_mid_ratio": 1.8, "mb_mid_attack_ms": 15.0, "mb_mid_release_ms": 130.0, "mb_mid_makeup_db": 0.0,
        "mb_high_threshold": 0.75, "mb_high_ratio": 1.6, "mb_high_attack_ms": 8.0,  "mb_high_release_ms": 90.0,  "mb_high_makeup_db": 0.0,
        "saturation_drive": 0.0, "saturation_mode": "tape", "saturation_mix": 0.0,
        "mid_gain_db": 0.0, "side_gain_db": 0.0, "stereo_width_amount": 1.0,
        "use_stereo_enhancer": False, "enhancer_bass_mono_freq": 120.0, "haas_delay_ms": 0.0,
        "reverb_size": 0.1, "reverb_wet": 0.0,
        "limiter_ceiling": 0.891, "limiter_release_ms": 90.0,
    },
}
def get_preset(name: str) -> dict:
    if name not in MASTERING_PRESETS:
        raise KeyError(f"Preset '{name}' no existe. Válidos: {sorted(MASTERING_PRESETS)}")
    return dict(MASTERING_PRESETS[name])

def _band_rms_db(band: np.ndarray, n_frames: int = 32) -> float:
    mono = band.mean(axis=0) if band.ndim == 2 else band
    if mono.size == 0:
        return -60.0
    rms = float(np.sqrt(np.mean(mono ** 2)))
    return float(20.0 * np.log10(rms + 1e-9))

# ─── Compresor de banda ancha (single-band / "de un solo cuerpo") ─────────────
# BUGFIX: este compresor tenía sliders en la UI (Threshold/Ratio/Attack/
# Release/Makeup) que el backend nunca leía ni usaba ("compresor fantasma"):
# ni app.py declaraba los Query params comp_*, ni apply_mastering_chain() ni
# process_audio() los recibían en su firma. Ahora se implementa la función y
# se conecta en la cadena de mastering (ver apply_mastering_chain, paso 4).
# Incluye oversampling 4x antes del cálculo de envolvente/ganancia para
# reducir aliasing en attacks rápidos, igual que harmonic_saturation().

def compressor(audio: np.ndarray, sr: int,
               threshold: float = 0.5,
               ratio: float = 4.0,
               attack_ms: float = 10.0,
               release_ms: float = 100.0,
               makeup_db: float = 0.0,
               oversample: int = DEFAULT_DSP_OVERSAMPLE,
               stereo_link: bool = True,
               return_gain_curve: bool = False) -> tuple:
    """Compresor feed-forward con soft-knee, make-up gain y link estéreo opcional.

    threshold: umbral lineal 0..1 (se convierte internamente a dBFS).
    ratio: relación de compresión, ej. 4.0 = "4:1".
    oversample: factor de sobremuestreo para detector/ganancia.
    stereo_link: usa una sola curva de ganancia para L/R y preserva el panorama.
    return_gain_curve: devuelve también la curva real de reducción para meters.
    """
    in_db = _band_rms_db(audio)
    if ratio <= 1.0:
        out = audio * (10.0 ** (makeup_db / 20.0))
        meter = {"gr_db": 0.0, "in_db": round(in_db, 2), "out_db": round(_band_rms_db(out), 2), "stereo_link": bool(stereo_link)}
        gr_shape = audio.shape if audio.ndim > 1 else (audio.shape[-1],)
        gr_arr = np.zeros(gr_shape, dtype=np.float64)
        return (out, meter, gr_arr) if return_gain_curve else (out, meter)

    threshold_db = 20.0 * np.log10(max(threshold, 1e-9))
    ovs = max(1, int(oversample))

    def gain_reduction(abs_signal):
        if ovs > 1:
            sr_up = sr * ovs
            up = resample_poly(abs_signal, ovs, 1)
            env = _smooth_envelope(np.abs(up), sr_up, attack_ms, release_ms)
            env_db = 20.0 * np.log10(env + 1e-9)
            if HAS_NUMBA:
                gr_db_up = _compute_gain_reduction_numba(env_db, threshold_db, ratio)
            else:
                gr_db_up = _soft_knee_gain_reduction_np(env_db, threshold_db, ratio)
            return resample_poly(gr_db_up, 1, ovs)[:len(abs_signal)]
        env = _smooth_envelope(np.abs(abs_signal), sr, attack_ms, release_ms)
        env_db = 20.0 * np.log10(env + 1e-9)
        if HAS_NUMBA:
            return _compute_gain_reduction_numba(env_db, threshold_db, ratio)
        return _soft_knee_gain_reduction_np(env_db, threshold_db, ratio)

    makeup = 10.0 ** (makeup_db / 20.0)

    if audio.ndim == 1:
        gr_arr = gain_reduction(np.abs(audio))
        out = audio * (10.0 ** (gr_arr / 20.0)) * makeup
    elif stereo_link:
        linked_detector = np.max(np.abs(audio), axis=0)
        linked_gr = gain_reduction(linked_detector)
        gr_arr = np.tile(linked_gr, (audio.shape[0], 1))
        out = audio * (10.0 ** (linked_gr / 20.0))[np.newaxis, :] * makeup
    else:
        grs = [gain_reduction(np.abs(ch)) for ch in audio]
        gr_arr = np.stack(grs)
        out = audio * (10.0 ** (gr_arr / 20.0)) * makeup

    tail = max(1, sr // 8)
    gr_db_mean = float(np.mean(gr_arr[..., -tail:])) if gr_arr.size else 0.0
    out_db = _band_rms_db(out)

    meter = {
        "gr_db": round(gr_db_mean, 2),
        "in_db": round(in_db, 2),
        "out_db": round(out_db, 2),
        "stereo_link": bool(stereo_link),
        "oversample": ovs,
    }
    return (out, meter, gr_arr) if return_gain_curve else (out, meter)

# ─── Compresor multibanda (solo se usa si se activa) ─────────────────────────

def multiband_compressor(audio: np.ndarray, sr: int,
                         low_crossover: float = 250.0,
                         high_crossover: float = 4000.0,
                         low_threshold: float = 0.7,
                         low_ratio: float = 2.0,
                         low_attack_ms: float = 20.0,
                         low_release_ms: float = 150.0,
                         low_makeup_db: float = 0.0,
                         mid_threshold: float = 0.7,
                         mid_ratio: float = 2.0,
                         mid_attack_ms: float = 20.0,
                         mid_release_ms: float = 150.0,
                         mid_makeup_db: float = 0.0,
                         high_threshold: float = 0.7,
                         high_ratio: float = 2.0,
                         high_attack_ms: float = 20.0,
                         high_release_ms: float = 150.0,
                         high_makeup_db: float = 0.0,
                         bypass: bool = True,
                         oversample: int = DEFAULT_DSP_OVERSAMPLE) -> tuple:
    if bypass:
        return audio, {"low_gr_db": 0.0, "mid_gr_db": 0.0, "high_gr_db": 0.0,
                       "low_in_db": 0.0, "mid_in_db": 0.0, "high_in_db": 0.0,
                       "low_out_db": 0.0, "mid_out_db": 0.0, "high_out_db": 0.0}

    if audio.ndim == 1:
        audio = np.stack([audio, audio])
    elif audio.shape[0] == 1:
        audio = np.stack([audio[0], audio[0]])

    low_crossover  = float(np.clip(low_crossover,  20.0, sr / 2.0 - 1.0))
    high_crossover = float(np.clip(high_crossover, low_crossover + 1.0, sr / 2.0 - 1.0))

    sos_lo_lp  = butter(4, low_crossover,  btype='lowpass',  fs=sr, output='sos')
    sos_lo_hp  = butter(4, low_crossover,  btype='highpass', fs=sr, output='sos')
    sos_hi_lp  = butter(4, high_crossover, btype='lowpass',  fs=sr, output='sos')
    sos_hi_hp  = butter(4, high_crossover, btype='highpass', fs=sr, output='sos')

    # Usar sosfiltfilt para fase cero
    low      = sosfiltfilt(sos_lo_lp, audio)
    mid_high = sosfiltfilt(sos_lo_hp, audio)
    mid      = sosfiltfilt(sos_hi_lp, mid_high)
    high     = sosfiltfilt(sos_hi_hp, mid_high)

    low_in_db  = _band_rms_db(low)
    mid_in_db  = _band_rms_db(mid)
    high_in_db = _band_rms_db(high)

    def _compressor(ch, threshold, ratio, attack_ms, release_ms, makeup_db):
        compressed, _meter, gr_db = compressor(
            ch, sr,
            threshold=threshold, ratio=ratio,
            attack_ms=attack_ms, release_ms=release_ms,
            makeup_db=makeup_db, oversample=oversample,
            stereo_link=True, return_gain_curve=True,
        )
        return compressed, gr_db

    low_comp,  low_gr_arr  = _compressor(low,  low_threshold,  low_ratio,  low_attack_ms,  low_release_ms,  low_makeup_db)
    mid_comp,  mid_gr_arr  = _compressor(mid,  mid_threshold,  mid_ratio,  mid_attack_ms,  mid_release_ms,  mid_makeup_db)
    high_comp, high_gr_arr = _compressor(high, high_threshold, high_ratio, high_attack_ms, high_release_ms, high_makeup_db)

    tail = max(1, sr // 8)
    low_gr_db  = float(np.mean(low_gr_arr[..., -tail:]))
    mid_gr_db  = float(np.mean(mid_gr_arr[..., -tail:]))
    high_gr_db = float(np.mean(high_gr_arr[..., -tail:]))

    low_out_db  = _band_rms_db(low_comp)
    mid_out_db  = _band_rms_db(mid_comp)
    high_out_db = _band_rms_db(high_comp)

    meter_data = {
        "low_gr_db":   round(low_gr_db, 2),
        "mid_gr_db":   round(mid_gr_db, 2),
        "high_gr_db":  round(high_gr_db, 2),
        "low_in_db":   round(low_in_db, 2),
        "mid_in_db":   round(mid_in_db, 2),
        "high_in_db":  round(high_in_db, 2),
        "low_out_db":  round(low_out_db, 2),
        "mid_out_db":  round(mid_out_db, 2),
        "high_out_db": round(high_out_db, 2),
    }

    return low_comp + mid_comp + high_comp, meter_data

# ─── Cadena DSP principal ──────────────────────────────────────────────────────
# Orden de la cadena (HPF, banda ancha, glue y limiter siempre activos;
# el resto es opcional según los parámetros del preset/usuario):
#   1. High-pass (limpieza de sub-graves, fase cero)
#   2. EQ paramétrico (4 bandas RBJ) + high shelf
#   3. Transient shaper
#   4. Dinámica, en orden: multibanda (bypass por defecto) → banda ancha /
#      "un solo cuerpo" (Threshold/Ratio/Attack/Release/Makeup, siempre
#      activo) → glue compressor (bypass por defecto, cierra la dinámica)
#   5. Saturación armónica (tape/tube, oversampling x4)
#   6. Imagen estéreo: Mid/Side gain → enhancer/width → multibanda estéreo
#   7. Reverb (efecto de espacio, sutil)
#   8. Limitador brick-wall con lookahead (siempre activo, último en la cadena)
# No se aplica ninguna normalización de ganancia en ningún punto.

def apply_mastering_chain(
    audio: np.ndarray, sr: int,
    target_peak: float = 0.95,          # No se usa
    use_lufs_normalize: bool = False,   # No se usa
    target_lufs: float = -14.0,         # No se usa
    input_gain_db: float = 0.0,
    oversample_mode: str = "quality",
    comp_stereo_link: bool = True,
    comp_threshold: float = 0.5,
    comp_ratio: float = 4.0,
    comp_attack_ms: float = 10.0,
    comp_release_ms: float = 100.0,
    comp_makeup_db: float = 0.0,
    mb_low_crossover: float = 250.0,
    mb_high_crossover: float = 4000.0,
    mb_low_threshold: float = 0.7,
    mb_low_ratio: float = 2.0,
    mb_low_attack_ms: float = 20.0,
    mb_low_release_ms: float = 150.0,
    mb_low_makeup_db: float = 0.0,
    mb_mid_threshold: float = 0.7,
    mb_mid_ratio: float = 2.0,
    mb_mid_attack_ms: float = 20.0,
    mb_mid_release_ms: float = 150.0,
    mb_mid_makeup_db: float = 0.0,
    mb_high_threshold: float = 0.7,
    mb_high_ratio: float = 2.0,
    mb_high_attack_ms: float = 20.0,
    mb_high_release_ms: float = 150.0,
    mb_high_makeup_db: float = 0.0,
    mb_bypass: bool = True,  # <-- AHORA DESACTIVADO POR DEFECTO
    hp_cutoff: float = 80.0,
    high_shelf_gain_db: float = 0.0,
    high_shelf_freq_hz: float = 8000.0,
    mb_stereo_bypass: bool = True,
    mb_stereo_low_width: float = 0.9,
    mb_stereo_mid_width: float = 1.2,
    mb_stereo_high_width: float = 1.5,
    mb_stereo_low_crossover: float = 150.0,
    mb_stereo_high_crossover: float = 4000.0,
    eq1_freq: float = 100.0, eq1_gain: float = 0.0, eq1_q: float = 1.0,
    eq2_freq: float = 500.0, eq2_gain: float = 0.0, eq2_q: float = 1.0,
    eq3_freq: float = 2000.0, eq3_gain: float = 0.0, eq3_q: float = 1.0,
    eq4_freq: float = 8000.0, eq4_gain: float = 0.0, eq4_q: float = 1.0,
    transient_attack: float = 0.0,
    transient_sustain: float = 0.0,
    saturation_drive: float = 0.0,
    saturation_mode: str = "tape",
    saturation_mix: float = 1.0,
    mid_gain_db: float = 0.0,
    side_gain_db: float = 0.0,
    stereo_width_amount: float = 1.0,
    use_stereo_enhancer: bool = False,
    enhancer_bass_mono_freq: float = 120.0,
    haas_delay_ms: float = 0.0,
    reverb_size: float = 0.3,
    reverb_wet: float = 0.0,
    glue_bypass: bool = True,
    glue_threshold_db: float = -4.0,
    glue_ratio: float = 2.0,
    glue_attack_ms: float = 30.0,
    glue_release_ms: float = 120.0,
    glue_makeup_db: float = 0.0,
    limiter_ceiling: float = 0.95,
    limiter_release_ms: float = 80.0,
    eq_mode: str = "iir",              # "iir" (zero-phase, actual) | "linear_phase" (FIR)
    linear_phase_taps: int = 2049,
    low_end_mono_freq: float = 120.0,
    low_end_mono_amount: float = 0.0,  # 0 = bypass (comportamiento anterior sin cambios)
    dyneq_bypass: bool = True,
    dyneq_freq: float = 3000.0,
    dyneq_q: float = 2.5,
    dyneq_threshold_db: float = -18.0,
    dyneq_ratio: float = 3.0,
    dyneq_attack_ms: float = 3.0,
    dyneq_release_ms: float = 80.0,
    dyneq_max_reduction_db: float = 12.0,
    nr_bypass: bool = True,
    nr_strength: float = 0.5,
    nr_noise_sample_sec: float = 0.5,
    progress_cb=None,
    progress_range: tuple = (0, 100),
    **_ignored,
) -> tuple:
    """
    Cadena de mastering (orden profesional estándar):

        input gain → HPF → EQ correctiva/tonal → transient shaper →
        DINÁMICA (multibanda → banda ancha → glue) → saturación armónica →
        imagen estéreo (mid/side → width/enhancer → multibanda estéreo) →
        reverb → limitador (siempre al final).

    La lógica es: primero se limpia y ecualiza la señal (correctivo), luego
    se le da forma a los transientes ANTES de comprimir (si no, el
    compresor ya alteró lo que el transient shaper querría tocar). Después
    va TODA la sección de dinámica junta y en el orden correcto: multibanda
    primero (equilibra cada banda de frecuencia por separado), banda ancha
    después (controla el nivel general ya balanceado) y glue al final de la
    dinámica (cohesiona el conjunto — por eso se llama "glue", pega la
    mezcla; no tiene sentido aplicarlo después de reverb/estéreo, tiene que
    ser parte de la etapa de dinámica). Con la señal ya balanceada en nivel
    se agrega color armónico (saturación), luego se trabaja la imagen
    estéreo, y recién ahí se suma el reverb (efecto de espacio, va casi al
    final para no ensuciar las etapas de dinámica/saturación previas). El
    limitador brick-wall siempre cierra la cadena. No se aplica ninguna
    normalización de ganancia automática (solo el trim manual de
    input_gain_db, si se especifica).
    """
    ovs = resolve_oversample(oversample_mode)

    # Reescala un 0-100 "local" de la cadena al rango [progress_range[0], progress_range[1]]
    # que le corresponde dentro del progreso total reportado por process_audio().
    _p_lo, _p_hi = progress_range
    def _chain_progress(local_pct: float, stage: str) -> None:
        _report(progress_cb, _p_lo + (local_pct / 100.0) * (_p_hi - _p_lo), stage)

    # ── 0a. Reducción de ruido (opcional, bypass por defecto) ─────────────
    # Va ANTES de todo el procesamiento: limpiar primero, procesar después.
    # Si está activa, reduce hiss/hum/ruido de sala estimando el perfil de
    # ruido de los primeros noise_sample_sec segundos del track.
    if not nr_bypass:
        _chain_progress(0, "Reduciendo ruido de fondo")
        audio = noise_reduction(audio, sr, strength=nr_strength,
                                noise_sample_sec=nr_noise_sample_sec)

    # ── 0. Input gain (trim manual, opcional) ──────────────────────────────
    _chain_progress(0, "Ajustando ganancia de entrada")
    if input_gain_db != 0.0:
        audio = audio * (10.0 ** (input_gain_db / 20.0))

    # ── 1. High-pass (fase cero) ──────────────────────────────────────────
    _chain_progress(5, "Aplicando filtro pasa-altos")
    audio = eq_high_pass(audio, sr, cutoff_hz=hp_cutoff)

    # ── 2. EQ paramétrico (4 bandas + high shelf, opcional banda por banda) ─
    _chain_progress(12, "Aplicando ecualización paramétrica")
    # Dos modos, elegidos por eq_mode:
    #   "iir"          -> como antes: cada banda es un biquad RBJ aplicado
    #                     con sosfiltfilt (fase cero, cascada banda a banda).
    #   "linear_phase" -> las bandas activas se combinan en UNA sola curva
    #                     de magnitud y se aplican con un único FIR de fase
    #                     lineal (linear_phase_eq), delay de grupo constante
    #                     y verificable en toda la banda.
    if str(eq_mode).lower() == "linear_phase":
        lp_bands = []
        for freq, gain, q in [
            (eq1_freq, eq1_gain, eq1_q),
            (eq2_freq, eq2_gain, eq2_q),
            (eq3_freq, eq3_gain, eq3_q),
            (eq4_freq, eq4_gain, eq4_q),
        ]:
            if gain != 0.0:
                lp_bands.append({"type": "peak", "freq": freq, "gain_db": gain, "q": q})
        if high_shelf_gain_db != 0.0:
            lp_bands.append({"type": "high_shelf", "freq": high_shelf_freq_hz, "gain_db": high_shelf_gain_db})
        if lp_bands:
            audio = linear_phase_eq(audio, sr, lp_bands, num_taps=linear_phase_taps)
    else:
        for freq, gain, q in [
            (eq1_freq, eq1_gain, eq1_q),
            (eq2_freq, eq2_gain, eq2_q),
            (eq3_freq, eq3_gain, eq3_q),
            (eq4_freq, eq4_gain, eq4_q),
        ]:
            if gain != 0.0:
                audio = eq_parametric_band(audio, sr, freq=freq, gain_db=gain, q=q)

        # High shelf (opcional) — freq variable
        if high_shelf_gain_db != 0.0:
            audio = eq_high_shelf(audio, sr, cutoff_hz=high_shelf_freq_hz, gain_db=high_shelf_gain_db)

    # ── 2b. Dynamic EQ de banda única (opcional, bypass por defecto) ───────
    # Va ACÁ (después del EQ estático, antes de la dinámica de nivel) porque
    # es correctiva sobre el timbre/resonancias — igual que el resto de la
    # EQ — y tiene que actuar antes de que el compresor de banda ancha vea
    # la señal, si no la resonancia ya afectó la detección de nivel.
    audio, dyneq_meters = dynamic_eq_band(
        audio, sr,
        freq=dyneq_freq, q=dyneq_q,
        threshold_db=dyneq_threshold_db, ratio=dyneq_ratio,
        attack_ms=dyneq_attack_ms, release_ms=dyneq_release_ms,
        max_reduction_db=dyneq_max_reduction_db, bypass=dyneq_bypass,
    )

    # ── 3. Transient shaper (opcional, ANTES de comprimir) ─────────────────
    _chain_progress(20, "Dando forma a los transientes")
    if transient_attack != 0.0 or transient_sustain != 0.0:
        audio = transient_shaper(audio, sr,
                                 attack_amount=transient_attack,
                                 sustain_amount=transient_sustain)

    # ══ DINÁMICA (multibanda → banda ancha → glue, todo junto y en ese ══
    # ══ orden: primero se equilibra cada banda, después el nivel      ══
    # ══ general, y al final el glue cohesiona el conjunto)            ══

    # ── 4. Compresor multibanda (bypass por defecto) ───────────────────────
    _chain_progress(30, "Aplicando compresión multibanda")
    audio, mb_meters = multiband_compressor(
        audio, sr,
        low_crossover=mb_low_crossover,
        high_crossover=mb_high_crossover,
        low_threshold=mb_low_threshold,
        low_ratio=mb_low_ratio,
        low_attack_ms=mb_low_attack_ms,
        low_release_ms=mb_low_release_ms,
        low_makeup_db=mb_low_makeup_db,
        mid_threshold=mb_mid_threshold,
        mid_ratio=mb_mid_ratio,
        mid_attack_ms=mb_mid_attack_ms,
        mid_release_ms=mb_mid_release_ms,
        mid_makeup_db=mb_mid_makeup_db,
        high_threshold=mb_high_threshold,
        high_ratio=mb_high_ratio,
        high_attack_ms=mb_high_attack_ms,
        high_release_ms=mb_high_release_ms,
        high_makeup_db=mb_high_makeup_db,
        bypass=mb_bypass,
        oversample=ovs,
    )

    # ── 5. Compresor de banda ancha ("un solo cuerpo": Threshold/Ratio/    ──
    # ── Attack/Release/Makeup). Antes era un control fantasma: la UI       ──
    # ── mandaba comp_threshold/comp_ratio/etc. pero ningún parámetro de    ──
    # ── esta función los recibía. Ahora siempre se aplica (como el resto  ──
    # ── de la sección "Dinámica" en la UI, que no tiene bypass propio).   ──
    _chain_progress(45, "Aplicando compresión de banda ancha")
    audio, comp_meters = compressor(
        audio, sr,
        threshold=comp_threshold,
        ratio=comp_ratio,
        attack_ms=comp_attack_ms,
        release_ms=comp_release_ms,
        makeup_db=comp_makeup_db,
        oversample=ovs,
        stereo_link=comp_stereo_link,
    )

    # ── 6. Glue compressor (opcional, cierra la sección de dinámica) ──────
    _chain_progress(55, "Aplicando glue compressor")
    glue_meters = {"bypass": True, "gr_db": 0.0}
    if not glue_bypass:
        audio, glue_meters = compressor(
            audio, sr,
            threshold=10.0 ** (glue_threshold_db / 20.0),
            ratio=glue_ratio,
            attack_ms=glue_attack_ms,
            release_ms=glue_release_ms,
            makeup_db=glue_makeup_db,
            oversample=ovs,
            stereo_link=True,
        )
        glue_meters.update({"bypass": False, "threshold_db": round(glue_threshold_db, 2)})

    # ── 7. Saturación armónica (opcional, oversampling x4) ─────────────────
    _chain_progress(62, "Aplicando saturación armónica")
    if saturation_drive > 0.0:
        audio = harmonic_saturation(audio, drive=saturation_drive,
                                    mode=saturation_mode, mix=saturation_mix,
                                    oversample=ovs)

    # ══ IMAGEN ESTÉREO (mid/side → width/enhancer → multibanda estéreo) ══

    # ── 8. Mid/Side gain (opcional) ────────────────────────────────────────
    _chain_progress(70, "Procesando imagen estéreo")
    if audio.shape[0] == 2 and (mid_gain_db != 0.0 or side_gain_db != 0.0):
        audio = mid_side_process(audio, mid_gain_db=mid_gain_db, side_gain_db=side_gain_db)

    # ── 8b. Estéreo: enhancer o width simple (opcional) ─────────────────────
    if audio.shape[0] == 2:
        if use_stereo_enhancer:
            audio = stereo_enhancer(audio, sr, width=stereo_width_amount,
                                    bass_mono_freq=enhancer_bass_mono_freq,
                                    haas_delay_ms=haas_delay_ms)
        elif stereo_width_amount != 1.0:
            audio = stereo_width(audio, width=stereo_width_amount)

    # ── 8b2. Low-End Mono Maker DEDICADO (opcional, bypass por defecto) ────
    # Independiente del bass_mono_freq fijo que ya trae stereo_enhancer:
    # este permite mono parcial (0..1) y se puede usar aunque no se use el
    # enhancer (p.ej. con stereo_width simple).
    if audio.shape[0] == 2 and low_end_mono_amount > 0.0:
        audio = low_end_mono_maker(audio, sr, freq=low_end_mono_freq, mono_amount=low_end_mono_amount)

    # ── 8c. Multiband Stereo Width (opcional, bypass por defecto) ──────────
    if audio.shape[0] == 2 and not mb_stereo_bypass:
        audio = multiband_stereo_width(
            audio, sr,
            low_width=mb_stereo_low_width,
            mid_width=mb_stereo_mid_width,
            high_width=mb_stereo_high_width,
            low_crossover=mb_stereo_low_crossover,
            high_crossover=mb_stereo_high_crossover,
        )

    # ── 9. Reverb (opcional, efecto de espacio — casi al final) ────────────
    _chain_progress(80, "Aplicando reverb")
    if reverb_wet > 0.0:
        audio = reverb_simple(audio, sr, room_size=reverb_size, wet=reverb_wet)

    # ---- SIN NORMALIZACIÓN ----

    # VU pre-limiter (para métricas)
    mono_pre = audio.mean(axis=0) if audio.ndim == 2 else audio
    pre_rms_db  = float(20.0 * np.log10(np.sqrt(np.mean(mono_pre ** 2)) + 1e-9))
    pre_peak_db = float(20.0 * np.log10(np.max(np.abs(mono_pre)) + 1e-9))

    # ── 10. Limitador brick-wall con lookahead (siempre activo, al final) ─
    _chain_progress(90, "Aplicando limitador final")
    audio = limiter(audio, sr, ceiling=limiter_ceiling, release_ms=limiter_release_ms, lookahead_ms=5.0,
                    oversample=ovs)

    # VU post-limiter
    mono_post = audio.mean(axis=0) if audio.ndim == 2 else audio
    post_rms_db  = float(20.0 * np.log10(np.sqrt(np.mean(mono_post ** 2)) + 1e-9))
    post_peak_db = float(20.0 * np.log10(np.max(np.abs(mono_post)) + 1e-9))
    post_lufs    = measure_lufs_integrated(audio, sr)
    post_corr    = stereo_correlation(audio)

    chain_meters = {
        "config": {"oversample": ovs, "oversample_mode": str(oversample_mode), "comp_stereo_link": bool(comp_stereo_link)},
        "comp": comp_meters,
        "glue": glue_meters,
        "mb": mb_meters,
        "dyneq": dyneq_meters,
        "pre_limiter":  {"rms_db": round(pre_rms_db, 2),  "peak_db": round(pre_peak_db, 2)},
        "post_limiter": {
            "rms_db":             round(post_rms_db, 2),
            "peak_db":            round(post_peak_db, 2),
            "lufs":               round(post_lufs, 2),
            "stereo_correlation": round(post_corr, 3),
        },
    }

    _chain_progress(100, "Cadena de mastering completa")
    return audio, chain_meters

# ─── Pipeline principal ────────────────────────────────────────────────────────

def process_audio(
    input_path: str,
    target_peak: float = 0.95,          # Ignorado
    use_lufs_normalize: bool = False,   # Ignorado
    target_lufs: float = -14.0,         # Ignorado
    input_gain_db: float = 0.0,
    oversample_mode: str = "quality",
    comp_stereo_link: bool = True,
    comp_threshold: float = 0.5,
    comp_ratio: float = 4.0,
    comp_attack_ms: float = 10.0,
    comp_release_ms: float = 100.0,
    comp_makeup_db: float = 0.0,
    mb_low_crossover: float = 250.0,
    mb_high_crossover: float = 4000.0,
    mb_low_threshold: float = 0.7,
    mb_low_ratio: float = 2.0,
    mb_low_attack_ms: float = 20.0,
    mb_low_release_ms: float = 150.0,
    mb_low_makeup_db: float = 0.0,
    mb_mid_threshold: float = 0.7,
    mb_mid_ratio: float = 2.0,
    mb_mid_attack_ms: float = 20.0,
    mb_mid_release_ms: float = 150.0,
    mb_mid_makeup_db: float = 0.0,
    mb_high_threshold: float = 0.7,
    mb_high_ratio: float = 2.0,
    mb_high_attack_ms: float = 20.0,
    mb_high_release_ms: float = 150.0,
    mb_high_makeup_db: float = 0.0,
    mb_bypass: bool = True,  # <-- DESACTIVADO POR DEFECTO
    hp_cutoff: float = 80.0,
    high_shelf_gain_db: float = 0.0,
    high_shelf_freq_hz: float = 8000.0,
    mb_stereo_bypass: bool = True,
    mb_stereo_low_width: float = 0.9,
    mb_stereo_mid_width: float = 1.2,
    mb_stereo_high_width: float = 1.5,
    mb_stereo_low_crossover: float = 150.0,
    mb_stereo_high_crossover: float = 4000.0,
    eq1_freq: float = 100.0, eq1_gain: float = 0.0, eq1_q: float = 1.0,
    eq2_freq: float = 500.0, eq2_gain: float = 0.0, eq2_q: float = 1.0,
    eq3_freq: float = 2000.0, eq3_gain: float = 0.0, eq3_q: float = 1.0,
    eq4_freq: float = 8000.0, eq4_gain: float = 0.0, eq4_q: float = 1.0,
    transient_attack: float = 0.0,
    transient_sustain: float = 0.0,
    saturation_drive: float = 0.0,
    saturation_mode: str = "tape",
    saturation_mix: float = 1.0,
    mid_gain_db: float = 0.0,
    side_gain_db: float = 0.0,
    stereo_width_amount: float = 1.0,
    use_stereo_enhancer: bool = False,
    enhancer_bass_mono_freq: float = 120.0,
    haas_delay_ms: float = 0.0,
    reverb_size: float = 0.3,
    reverb_wet: float = 0.0,
    glue_bypass: bool = True,
    glue_threshold_db: float = -4.0,
    glue_ratio: float = 2.0,
    glue_attack_ms: float = 30.0,
    glue_release_ms: float = 120.0,
    glue_makeup_db: float = 0.0,
    limiter_ceiling: float = 0.95,
    limiter_release_ms: float = 80.0,
    eq_mode: str = "iir",
    linear_phase_taps: int = 2049,
    low_end_mono_freq: float = 120.0,
    low_end_mono_amount: float = 0.0,
    dyneq_bypass: bool = True,
    dyneq_freq: float = 3000.0,
    dyneq_q: float = 2.5,
    dyneq_threshold_db: float = -18.0,
    dyneq_ratio: float = 3.0,
    dyneq_attack_ms: float = 3.0,
    dyneq_release_ms: float = 80.0,
    dyneq_max_reduction_db: float = 12.0,
    nr_bypass: bool = True,
    nr_strength: float = 0.5,
    nr_noise_sample_sec: float = 0.5,
    output_format: str = "wav",
    output_bit_depth: int = DEFAULT_OUTPUT_BIT_DEPTH,
    preview_seconds: float = None,
    platform_target: str = None,
    progress_cb=None,
) -> dict:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Archivo no encontrado: {input_path}")

    # Ajustar ceiling según plataforma
    if platform_target:
        target = get_platform_target(platform_target)
        limiter_ceiling = 10.0 ** (target["true_peak_db"] / 20.0)
        # No se usa normalización LUFS

    _report(progress_cb, 2, "Cargando archivo de audio")
    audio, sr = librosa.load(input_path, sr=None, mono=False)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]

    if preview_seconds is not None:
        audio = _crop_preview(audio, sr, preview_seconds)

    # Se guarda una copia del audio ANTES de la cadena: el LUFS safety check
    # (más abajo) necesita re-renderizar la cadena completa desde el audio
    # original con un input_gain_db corregido en cada iteración. Si se
    # reutilizara `audio` después de `apply_mastering_chain` se estaría
    # re-encadenando la cadena sobre su propia salida en cada intento.
    audio_orig = audio.copy()

    _report(progress_cb, 5, "Analizando audio original")
    analysis_before = analyze_audio(audio, sr)

    # MEJORA (fix #6): construir MasteringParams UNA sola vez desde los locals.
    # Tanto la llamada inicial como el safety check usan este objeto — el único
    # punto de verdad de todos los parámetros de la cadena.
    _chain_params = MasteringParams(
        input_gain_db=input_gain_db,
        target_peak=target_peak,
        use_lufs_normalize=use_lufs_normalize,
        target_lufs=target_lufs,
        oversample_mode=oversample_mode,
        hp_cutoff=hp_cutoff,
        eq_mode=eq_mode,
        linear_phase_taps=linear_phase_taps,
        high_shelf_gain_db=high_shelf_gain_db,
        high_shelf_freq_hz=high_shelf_freq_hz,
        eq1_freq=eq1_freq, eq1_gain=eq1_gain, eq1_q=eq1_q,
        eq2_freq=eq2_freq, eq2_gain=eq2_gain, eq2_q=eq2_q,
        eq3_freq=eq3_freq, eq3_gain=eq3_gain, eq3_q=eq3_q,
        eq4_freq=eq4_freq, eq4_gain=eq4_gain, eq4_q=eq4_q,
        dyneq_bypass=dyneq_bypass, dyneq_freq=dyneq_freq, dyneq_q=dyneq_q,
        dyneq_threshold_db=dyneq_threshold_db, dyneq_ratio=dyneq_ratio,
        dyneq_attack_ms=dyneq_attack_ms, dyneq_release_ms=dyneq_release_ms,
        dyneq_max_reduction_db=dyneq_max_reduction_db,
        transient_attack=transient_attack, transient_sustain=transient_sustain,
        mb_bypass=mb_bypass,
        mb_low_crossover=mb_low_crossover, mb_high_crossover=mb_high_crossover,
        mb_low_threshold=mb_low_threshold, mb_low_ratio=mb_low_ratio,
        mb_low_attack_ms=mb_low_attack_ms, mb_low_release_ms=mb_low_release_ms, mb_low_makeup_db=mb_low_makeup_db,
        mb_mid_threshold=mb_mid_threshold, mb_mid_ratio=mb_mid_ratio,
        mb_mid_attack_ms=mb_mid_attack_ms, mb_mid_release_ms=mb_mid_release_ms, mb_mid_makeup_db=mb_mid_makeup_db,
        mb_high_threshold=mb_high_threshold, mb_high_ratio=mb_high_ratio,
        mb_high_attack_ms=mb_high_attack_ms, mb_high_release_ms=mb_high_release_ms, mb_high_makeup_db=mb_high_makeup_db,
        comp_stereo_link=comp_stereo_link,
        comp_threshold=comp_threshold, comp_ratio=comp_ratio,
        comp_attack_ms=comp_attack_ms, comp_release_ms=comp_release_ms, comp_makeup_db=comp_makeup_db,
        glue_bypass=glue_bypass, glue_threshold_db=glue_threshold_db, glue_ratio=glue_ratio,
        glue_attack_ms=glue_attack_ms, glue_release_ms=glue_release_ms, glue_makeup_db=glue_makeup_db,
        saturation_drive=saturation_drive, saturation_mode=saturation_mode, saturation_mix=saturation_mix,
        mid_gain_db=mid_gain_db, side_gain_db=side_gain_db, stereo_width_amount=stereo_width_amount,
        use_stereo_enhancer=use_stereo_enhancer, enhancer_bass_mono_freq=enhancer_bass_mono_freq,
        haas_delay_ms=haas_delay_ms,
        low_end_mono_freq=low_end_mono_freq, low_end_mono_amount=low_end_mono_amount,
        mb_stereo_bypass=mb_stereo_bypass,
        mb_stereo_low_width=mb_stereo_low_width, mb_stereo_mid_width=mb_stereo_mid_width,
        mb_stereo_high_width=mb_stereo_high_width,
        mb_stereo_low_crossover=mb_stereo_low_crossover, mb_stereo_high_crossover=mb_stereo_high_crossover,
        reverb_size=reverb_size, reverb_wet=reverb_wet,
        limiter_ceiling=limiter_ceiling, limiter_release_ms=limiter_release_ms,
        nr_bypass=nr_bypass, nr_strength=nr_strength, nr_noise_sample_sec=nr_noise_sample_sec,
    )
    audio, chain_meters = apply_mastering_chain(
        audio, sr,
        progress_cb=progress_cb, progress_range=(8, 80),
        **_chain_params.as_chain_kwargs(),
    )

    # ── LUFS safety check ──────────────────────────────────────────────────
    # MEJORA (fix #6): antes el safety check duplicaba manualmente los ~50
    # kwargs de apply_mastering_chain, lo que hacía que cualquier parámetro
    # nuevo agregado a la firma quedara silenciosamente ignorado en las
    # re-renderizaciones. Ahora se usa MasteringParams como único punto de
    # verdad: solo se actualiza input_gain_db y se pasa el dict completo,
    # eliminando la posibilidad de desincronización.
    lufs_safety_notes = []
    if use_lufs_normalize:
        max_iters = 4
        tolerance_db = 0.3
        current_input_gain = input_gain_db
        for i in range(max_iters):
            achieved = chain_meters.get("post_limiter", {}).get("lufs")
            if achieved is None:
                break
            delta = float(target_lufs) - float(achieved)
            if abs(delta) <= tolerance_db:
                lufs_safety_notes.append(
                    f"LUFS safety check: {achieved:.2f} LUFS vs. objetivo {target_lufs:.2f} LUFS "
                    f"(dentro de tolerancia, sin corrección adicional)."
                )
                break
            new_input_gain = float(np.clip(current_input_gain + delta, -24.0, 24.0))
            if abs(new_input_gain - current_input_gain) < 0.05:
                lufs_safety_notes.append(
                    f"LUFS safety check: no se pudo alcanzar {target_lufs:.2f} LUFS sin exceder "
                    f"el rango de input_gain_db (quedó en {achieved:.2f} LUFS)."
                )
                break
            current_input_gain = new_input_gain
            _report(progress_cb, 80 + i * 3,
                    f"Ajustando loudness (LUFS safety check, intento {i + 1}/{max_iters})")
            # Único punto de cambio: input_gain_db corregido. Todo lo demás
            # viene del dataclass — no hay lista manual de kwargs que
            # desincronizarse con la firma real de apply_mastering_chain.
            retry_kwargs = _chain_params.as_chain_kwargs()
            retry_kwargs["input_gain_db"] = current_input_gain
            audio_retry, chain_meters_retry = apply_mastering_chain(
                audio_orig, sr, progress_cb=None, **retry_kwargs
            )
            audio, chain_meters = audio_retry, chain_meters_retry
            lufs_safety_notes.append(
                f"LUFS safety check #{i + 1}: {achieved:.2f} LUFS vs. objetivo {target_lufs:.2f} LUFS "
                f"→ input_gain_db corregido a {current_input_gain:+.2f} dB."
            )
        chain_meters["lufs_safety"] = {
            "enabled": True,
            "target_lufs": round(float(target_lufs), 2),
            "final_input_gain_db": round(current_input_gain, 2),
            "notes": lufs_safety_notes,
        }
    else:
        chain_meters["lufs_safety"] = {"enabled": False}

    _report(progress_cb, 93, "Analizando resultado final")
    analysis_after = analyze_audio(audio, sr)

    base = os.path.splitext(os.path.basename(input_path))[0]
    suffix = "_preview" if preview_seconds else "_mastered"
    output_path = f"processed/{base}_{uuid.uuid4().hex[:8]}{suffix}.{output_format}"

    audio_out = audio[0] if audio.shape[0] == 1 else audio.T

    _report(progress_cb, 97, "Guardando archivo masterizado")
    effective_bit_depth = _write_master_output(
        audio_out, sr, output_path, output_format, output_bit_depth
    )

    _report(progress_cb, 100, "Mastering completado")
    return {
        "output_path":       output_path,
        "output_bit_depth":  effective_bit_depth,
        "analysis_before":   analysis_before,
        "analysis_after":    analysis_after,
        "mix_advice_before": mix_advice(analysis_before),
        "mix_advice_after":  mix_advice(analysis_after),
        "chain_meters":      chain_meters,
    }

# ─── Mastering por referencia (reference-track matching) ──────────────────────
# Toma un track de referencia (ya masterizado, del sonido que se quiere imitar)
# y adapta el track propio hacia ese objetivo en 4 dimensiones:
#   1. Balance tonal (EQ de "matching" multibanda, derivada de la diferencia
#      espectral entre ambos tracks)
#   2. Loudness (LUFS integrado)
#   3. Dinámica (crest factor: si el propio track es mucho más dinámico que
#      la referencia, se aplica un compresor de banda ancha suave para acercarlo)
#   4. Ancho estéreo (correlación L/R)
# y finalmente limita al techo de pico aproximado de la referencia.

def _load_audio_any(path: str):
    audio, sr = librosa.load(path, sr=None, mono=False)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    return audio, sr

def spectral_energy_at_bands(audio: np.ndarray, sr: int, band_edges: list) -> list:
    """Energía promedio (dB) en bandas de frecuencia arbitrarias (lo, hi) en Hz.
    A diferencia de spectrum_analysis_fft (que usa bandas log fijas 20Hz..nyquist
    propio de cada archivo), esta función recibe los mismos band_edges para dos
    archivos con distinto sample rate, permitiendo comparar "manzanas con manzanas".

    Es la base directa de `compute_reference_eq_curve`: los frames se muestrean
    con `_averaged_magnitude_spectrum`, distribuidos en TODO el archivo (ver esa
    función), no solo en los primeros segundos.
    """
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    n_fft = min(8192, len(mono))
    n_fft = max(256, 2 ** int(np.floor(np.log2(max(n_fft, 256)))))
    avg_mag = _averaged_magnitude_spectrum(mono, n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    edges = [band_edges[0][0]] + [hi for _, hi in band_edges]
    return _log_band_average_db(freqs, avg_mag, np.array(edges, dtype=float)).tolist()

def spectral_match_score(a_db: list, b_db: list) -> dict:
    """Puntaje de similitud espectral entre dos curvas de bandas (0-100%).
    Ignora el offset general de nivel (loudness), solo mide diferencias de
    'forma'/balance tonal.
    """
    a = np.array(a_db, dtype=np.float64)
    b = np.array(b_db, dtype=np.float64)
    diff = (a - b)
    diff = diff - np.mean(diff)
    mae = float(np.mean(np.abs(diff)))
    score = float(np.clip(100.0 - mae * 8.0, 0.0, 100.0))
    return {"mean_abs_diff_db": round(mae, 2), "match_percent": round(score, 1)}

def _soft_clip_curve(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Soft-knee clip (tanh) para una curva de ganancias en dB.

    A diferencia de np.clip (que genera un quiebre de pendiente instantáneo
    apenas la curva toca el límite), esta versión se comporta casi como
    identidad lejos de los límites y se aplana suavemente cerca de/superando
    `lo`/`hi`. Esos quiebres abruptos son justamente lo que firwin2 traduce
    en ringing/artefactos audibles al construir el FIR de matching, así que
    evitarlos mejora directamente qué tan "limpio" suena el resultado.
    """
    center = (hi + lo) / 2.0
    half_range = (hi - lo) / 2.0
    if half_range <= 1e-9:
        return np.clip(x, lo, hi)
    return center + np.tanh((x - center) / half_range) * half_range

def compute_reference_eq_curve(src_bands_db: list, ref_bands_db: list, freqs_hz: list,
                                max_boost_db: float = 6.0, max_cut_db: float = -9.0,
                                smooth_window: int = 3,
                                blend: float = 0.75) -> list:
    """Calcula la curva de EQ (freq, gain_db) necesaria para acercar el balance
    tonal de src hacia ref. Se resta la media global (el loudness se maneja
    aparte, vía LUFS) y se suaviza/clippea para evitar EQs extremas o ásperas.

    MEJORA (matching "áspero"/artificial): la versión anterior suavizaba con
    un promedio móvil rectangular muy débil (y ademas 'mode=same', que atenúa
    los extremos hacia cero por el padding implícito) y después recortaba con
    un clip duro. El resultado era una curva objetivo con micro-escalones y
    quiebres de pendiente que build_matching_fir (via firwin2) reproduce
    fielmente como ringing/coloración áspera — el FIR es literalmente tan
    "prolijo" como la curva que se le pide. Ahora:
      1) Se suaviza con un kernel gaussiano (con padding por reflexión, no
         ceros) cuyo ancho escala con la cantidad de bandas, para una curva
         continua sin perder resolución real.
      2) Se recorta con soft-knee (tanh) en vez de clip duro.
      3) Se re-centra después del recorte, porque un soft-knee asimétrico
         (max_boost != |max_cut|) puede introducir un pequeño remanente de
         nivel medio que de otro modo se solaparía con el ajuste de LUFS.
      4) Se aplica un blend y límites perceptuales por zona para evitar
         matches 100% demasiado artificiales, especialmente en sub y presencia.
    """
    src = np.array(src_bands_db, dtype=np.float64)
    ref = np.array(ref_bands_db, dtype=np.float64)
    n = len(src)
    diff = ref - src
    diff = diff - np.mean(diff)

    if smooth_window > 1 and n > 2:
        sigma = max(smooth_window / 2.5, 0.6)
        radius = min(n - 1, max(1, int(round(sigma * 3))))
        x = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
        padded = np.pad(diff, radius, mode="reflect")
        diff = np.convolve(padded, kernel, mode="valid")

    diff = _soft_clip_curve(diff, max_cut_db, max_boost_db)
    diff = diff - np.mean(diff)

    freqs = np.array(freqs_hz, dtype=np.float64)
    zone_boost = np.full_like(freqs, max_boost_db, dtype=np.float64)
    zone_cut = np.full_like(freqs, max_cut_db, dtype=np.float64)
    zone_boost[freqs < 80.0] = min(max_boost_db, 3.0)
    zone_cut[freqs < 80.0] = max(max_cut_db, -6.0)
    presence = (freqs >= 2500.0) & (freqs <= 7000.0)
    zone_boost[presence] = min(max_boost_db, 3.5)
    zone_cut[presence] = max(max_cut_db, -6.0)
    air = freqs >= 10000.0
    zone_boost[air] = min(max_boost_db, 5.0)
    zone_cut[air] = max(max_cut_db, -7.0)

    blend = float(np.clip(blend, 0.0, 1.0))
    diff = np.clip(diff * blend, zone_cut, zone_boost)

    return [(float(f), float(g)) for f, g in zip(freqs_hz, diff.tolist())]

def apply_reference_eq_curve(audio: np.ndarray, sr: int, curve: list,
                             q: float = 1.1, min_gain_db: float = 0.3) -> np.ndarray:
    """(Legacy) Aplica la curva de matching como una cascada de filtros
    paramétricos (peaking), uno por banda. Se mantiene por si se necesita un
    EQ 'de consola' clásico, pero para el matching automático se usa
    build_matching_fir/apply_matching_fir (ver abajo), que evita el problema
    de bandas log-espaciadas vecinas que se solapan y se refuerzan entre sí
    al aplicarse en cascada, produciendo overshoot en vez de la curva pedida.
    """
    out = audio
    for freq, gain in curve:
        if abs(gain) < min_gain_db:
            continue
        out = eq_parametric_band(out, sr, freq=freq, gain_db=gain, q=q)
    return out

def build_matching_fir(curve: list, sr: int, precision: float = 1.1) -> np.ndarray:
    """Diseña un filtro FIR de fase lineal que realiza la curva de matching
    (freq_hz, gain_db) directamente en el dominio de la frecuencia, vía
    scipy.signal.firwin2. A diferencia de encadenar N filtros paramétricos
    (que se solapan/interfieren entre sí, especialmente con bandas log
    espaciadas cercanas), esto aplica la respuesta de frecuencia objetivo tal
    cual, banda por banda, sin refuerzo cruzado — el enfoque estándar de los
    plugins de "spectral match EQ".
    `precision` escala la cantidad de taps (resolución en frecuencia/octava);
    valores más altos = filtro más preciso/angosto pero más costoso.
    """
    from scipy.signal import firwin2
    if not curve:
        return None
    nyq = sr / 2.0
    freqs = np.clip(np.array([f for f, _ in curve], dtype=np.float64), 1.0, nyq - 1.0)
    gains_db = np.array([g for _, g in curve], dtype=np.float64)
    # Extiende con puntos planos en 0 Hz y en Nyquist para que firwin2 tenga
    # un dominio completo 0..nyquist bien definido.
    freqs_ext = np.concatenate(([0.0], freqs, [nyq]))
    gains_ext = np.concatenate(([gains_db[0]], gains_db, [gains_db[-1]]))
    for i in range(1, len(freqs_ext)):
        if freqs_ext[i] <= freqs_ext[i - 1]:
            freqs_ext[i] = freqs_ext[i - 1] + 1e-3
    freqs_norm = np.clip(freqs_ext / nyq, 0.0, 1.0)
    gains_lin = 10.0 ** (gains_ext / 20.0)
    numtaps = int(np.clip(2048 * precision, 511, 8191))
    if numtaps % 2 == 0:
        numtaps += 1
    return firwin2(numtaps, freqs_norm, gains_lin)

def apply_matching_fir(audio: np.ndarray, sr: int, taps) -> np.ndarray:
    """Aplica el FIR de matching con convolución 'same' (fftconvolve), que
    para un FIR de fase lineal compensa exactamente el delay de grupo,
    quedando efectivamente en fase cero."""
    if taps is None or len(taps) == 0:
        return audio
    if audio.ndim == 1:
        return fftconvolve(audio, taps, mode="same")
    return np.stack([fftconvolve(ch, taps, mode="same") for ch in audio])

# ─── Dinámica multibanda por referencia ────────────────────────────────────────
# BUGFIX/MEJORA: la versión anterior de "match_dynamics" comprimía TODA la
# señal con un único compresor de banda ancha, calculando un solo crest
# factor global. Eso podía, por ejemplo, aplastar los graves para igualar un
# crest factor global aunque los graves ya estuvieran igual de comprimidos
# que la referencia (y viceversa con los agudos). Ahora se compara y corrige
# banda por banda (graves/medios/agudos), igual que haría un ingeniero
# revisando cada rango por separado.

def match_dynamics_bands(audio: np.ndarray, sr: int,
                         own_crest: dict, ref_crest: dict,
                         bands: list = DYNAMICS_BANDS,
                         margin_db: float = 1.0,
                         oversample: int = DEFAULT_DSP_OVERSAMPLE) -> tuple:
    """Comprime banda por banda solo donde el crest factor propio supera al
    de la referencia por más de `margin_db`. Nunca expande dinámica (si la
    referencia es más dinámica que el track en una banda, se deja intacta
    para evitar artefactos de expansores)."""
    if audio.ndim == 1:
        audio2 = np.stack([audio, audio])
    else:
        audio2 = audio

    out = np.zeros_like(audio2)
    meta = {}
    attack_by_band  = {"low": 25.0, "mid": 15.0, "high": 8.0}
    release_by_band = {"low": 160.0, "mid": 110.0, "high": 70.0}

    for name, lo, hi in bands:
        band_audio = _bandpass_filter(audio2, sr, lo, hi)
        gap = own_crest.get(name, 0.0) - ref_crest.get(name, 0.0)
        if gap > margin_db:
            ratio = float(np.clip(1.4 + gap / 5.0, 1.4, 4.5))
            mono_b = band_audio.mean(axis=0)
            b_rms_db = float(20.0 * np.log10(np.sqrt(np.mean(mono_b ** 2)) + 1e-9))
            threshold_db  = b_rms_db + 3.0
            threshold_lin = float(np.clip(10.0 ** (threshold_db / 20.0), 0.02, 0.95))
            band_out, band_meter = compressor(
                band_audio, sr, threshold=threshold_lin, ratio=ratio,
                attack_ms=attack_by_band[name], release_ms=release_by_band[name],
                makeup_db=0.0, oversample=oversample,
            )
            out += band_out
            meta[name] = {"applied": True, "ratio": round(ratio, 2), "gap_db": round(gap, 2), **band_meter}
        else:
            out += band_audio
            meta[name] = {"applied": False, "gap_db": round(gap, 2)}

    if audio.ndim == 1:
        out = out.mean(axis=0)
    return out, meta

def match_lra(audio: np.ndarray, sr: int, own_lra: float, ref_lra: float,
             margin: float = 1.0,
             oversample: int = DEFAULT_DSP_OVERSAMPLE) -> tuple:
    """Ajusta la macro-dinámica (LRA, rango dinámico "a largo plazo") con un
    compresor tipo 'glue' (attack/release lentos) cuando el track propio es
    notablemente más variable en el tiempo que la referencia. Al igual que
    match_dynamics_bands, solo comprime, nunca expande."""
    meta = {"applied": False, "own_lra": round(own_lra, 2), "ref_lra": round(ref_lra, 2)}
    gap = own_lra - ref_lra
    if gap > margin:
        ratio = float(np.clip(1.2 + gap / 8.0, 1.2, 2.5))
        mono = audio.mean(axis=0) if audio.ndim == 2 else audio
        rms_db = float(20.0 * np.log10(np.sqrt(np.mean(mono ** 2)) + 1e-9))
        threshold_lin = float(np.clip(10.0 ** ((rms_db + 1.0) / 20.0), 0.05, 0.95))
        audio, comp_meter = compressor(audio, sr, threshold=threshold_lin, ratio=ratio,
                                       attack_ms=30.0, release_ms=300.0, makeup_db=0.0,
                                       oversample=oversample)
        meta.update({"applied": True, "ratio": round(ratio, 2), **comp_meter})
    return audio, meta

# ─── Estéreo multibanda por referencia ─────────────────────────────────────────
# BUGFIX/MEJORA: la versión anterior calculaba UNA sola correlación L/R para
# toda la señal y aplicaba un único factor de ancho global. Un master real
# casi siempre tiene graves mono/casi-mono y agudos más anchos: promediar
# todo en un solo número perdía esa forma y podía, por ejemplo, ensanchar los
# graves (rompiendo compatibilidad mono/fase) para igualar una correlación
# global dominada por los agudos. Ahora se hace banda por banda usando la
# relación cerrada entre la energía de Mid/Side y el coeficiente de
# correlación:
#   rho = (var(M) - var(S)) / (var(M) + var(S))
# despejando el factor de escala k a aplicar sobre Side para llegar a rho':
#   k = sqrt( var(M)*(1-rho') / (var(S)*(1+rho')) )

def match_stereo_bands(audio: np.ndarray, sr: int, target_corr: dict,
                       bands: list = DYNAMICS_BANDS, blend: float = 0.85,
                       min_k: float = 0.25, max_k: float = 2.5) -> tuple:
    if audio.ndim != 2 or audio.shape[0] != 2:
        return audio, {name: 1.0 for name, _, _ in bands}

    left, right = audio[0], audio[1]
    out_l = np.zeros_like(left)
    out_r = np.zeros_like(right)
    k_applied = {}

    for name, lo, hi in bands:
        fl = _bandpass_filter(left, sr, lo, hi)
        fr = _bandpass_filter(right, sr, lo, hi)
        mid  = (fl + fr) * 0.5
        side = (fl - fr) * 0.5
        var_m = float(np.var(mid))
        var_s = float(np.var(side))
        # Guarda de estabilidad: si el contenido "side" de esta banda es
        # prácticamente silencio (banda casi perfectamente mono, típico en
        # graves), var_s puede ser ~0. Escalar eso con la fórmula cerrada
        # amplificaría solo ruido de redondeo, no señal real, y podría
        # colorear el espectro de forma audible. En ese caso no se toca.
        if var_s < max(var_m, 1e-12) * 1e-4:
            out_l += mid + side
            out_r += mid - side
            k_applied[name] = 1.0
            continue
        cur_rho = float(np.clip((var_m - var_s) / (var_m + var_s + 1e-12), -1.0, 1.0))
        target_rho = float(np.clip(target_corr.get(name, cur_rho), -1.0, 1.0))
        blended_rho = float(np.clip(cur_rho + (target_rho - cur_rho) * blend, -0.98, 0.98))
        k2 = var_m * (1.0 - blended_rho) / (var_s * (1.0 + blended_rho) + 1e-12)
        k = float(np.clip(np.sqrt(max(k2, 0.0)), min_k, max_k))
        out_l += mid + side * k
        out_r += mid - side * k
        k_applied[name] = round(k, 3)

    return np.stack([out_l, out_r]), k_applied

# ─── Reporte / análisis inteligente del matching ───────────────────────────────

def reference_intelligent_report(tonal_after: dict, loudness_gain_db: float,
                                  dynamics_band_meta: dict, lra_meta: dict,
                                  stereo_k_applied: dict) -> dict:
    """Combina el resultado de las 4 dimensiones del matching (tonal,
    loudness, dinámica por banda + LRA, estéreo por banda) en un puntaje
    único y una lista de observaciones/consejos en lenguaje natural, similar
    a mix_advice() pero relativo a qué tan cerca quedó el track de la
    referencia y qué se le ajustó para lograrlo."""
    issues, tips = [], []
    score = 100.0

    tonal_pct = tonal_after.get("match_percent", 100.0)
    if tonal_pct < 60:
        issues.append(f"El match tonal final quedó en {tonal_pct}%: el balance de frecuencias todavía difiere bastante de la referencia.")
        tips.append("Probá subir eq_max_boost_db/eq_max_cut_db, o elegí una referencia de un género/instrumentación más parecida a tu track.")
        score -= 22
    elif tonal_pct < 80:
        tips.append(f"Match tonal de {tonal_pct}%: aceptable, pero todavía queda diferencia de timbre con la referencia.")
        score -= 10

    for name in ("low", "mid", "high"):
        meta = dynamics_band_meta.get(name, {})
        gap = meta.get("gap_db", 0.0)
        if meta.get("applied"):
            tips.append(f"Banda {name}: comprimida (~{gap} dB más dinámica que la referencia) para acercar la pegada de esa zona.")
            score -= min(8.0, abs(gap) * 0.6)
        elif gap < -3.0:
            tips.append(f"Banda {name}: ya es más densa/comprimida que la referencia; no se tocó para no perder dinámica de más.")

    if lra_meta.get("applied"):
        tips.append(f"Macro-dinámica (LRA) reducida de {lra_meta['own_lra']} a un valor más cercano a la referencia ({lra_meta['ref_lra']} LU) con compresión 'glue' suave.")
        score -= 6
    else:
        gap = lra_meta.get("own_lra", 0.0) - lra_meta.get("ref_lra", 0.0)
        if gap < -1.0:
            tips.append("Tu rango dinámico global (LRA) ya es menor que el de la referencia; no se aplicó compresión adicional de macro-dinámica.")

    for name, k in stereo_k_applied.items():
        if k > 1.08:
            tips.append(f"Estéreo banda {name}: ensanchado x{k} para acercarlo a la referencia.")
        elif k < 0.92:
            tips.append(f"Estéreo banda {name}: angostado/centrado x{k} para acercarlo a la referencia (evita problemas de fase/mono).")

    if abs(loudness_gain_db) > 0.3:
        tips.append(f"Loudness ajustado {loudness_gain_db:+.2f} dB para igualar el LUFS integrado de la referencia.")

    score = float(np.clip(score, 0.0, 100.0))
    grade = ("Excelente" if score >= 88 else "Buena" if score >= 72 else
             "Aceptable" if score >= 50 else "Necesita ajuste manual")
    if not tips:
        tips.append("El match con la referencia es muy sólido en las 4 dimensiones analizadas (tonal, loudness, dinámica y estéreo).")

    return {"overall_score": round(score, 1), "grade": grade, "issues": issues, "tips": tips}

def process_audio_with_reference(
    input_path: str,
    reference_path: str,
    eq_bands: int = 28,
    eq_max_boost_db: float = 6.0,
    eq_max_cut_db: float = -9.0,
    eq_q: float = 1.3,
    eq_match_blend: float = 0.75,
    oversample_mode: str = "quality",
    match_loudness: bool = True,
    match_dynamics: bool = True,
    match_stereo_width: bool = True,
    hp_cutoff: float = 30.0,
    limiter_release_ms: float = 60.0,
    output_format: str = "wav",
    output_bit_depth: int = DEFAULT_OUTPUT_BIT_DEPTH,
    preview_seconds: float = None,
    dynamics_margin_db: float = 1.0,
    stereo_blend: float = 0.85,
    progress_cb=None,
) -> dict:
    """Masteriza `input_path` adaptando su sonido al de `reference_path` en
    4 dimensiones: balance tonal (EQ de matching FIR), loudness (LUFS),
    dinámica (compresión multibanda + LRA) y ancho estéreo (matching de
    correlación L/R banda por banda). Incluye un reporte de análisis
    inteligente (`reference_intelligent_report`) que resume qué se ajustó y
    qué tan cerca quedó el resultado de la referencia."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Archivo no encontrado: {input_path}")
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Track de referencia no encontrado: {reference_path}")

    _report(progress_cb, 3, "Cargando audio propio y de referencia")
    audio, sr = _load_audio_any(input_path)
    ref_audio, ref_sr = _load_audio_any(reference_path)

    if preview_seconds is not None:
        audio = _crop_preview(audio, sr, preview_seconds)

    ovs = resolve_oversample(oversample_mode)

    _report(progress_cb, 8, "Analizando audio propio y de referencia")
    analysis_before = analyze_audio(audio, sr)
    analysis_reference = analyze_audio(ref_audio, ref_sr)

    # ── 1. Bandas comunes de frecuencia (log-spaced), acotadas al nyquist
    #        más chico de los dos archivos para poder compararlos entre sí ──
    nyquist = min(sr, ref_sr) / 2.0
    max_freq = float(np.clip(min(20000.0, nyquist - 100.0), 200.0, nyquist - 1.0))
    edges = np.logspace(np.log10(20.0), np.log10(max_freq), eq_bands + 1)
    band_edges = list(zip(edges[:-1].tolist(), edges[1:].tolist()))
    centers = [float(np.sqrt(lo * hi)) for lo, hi in band_edges]

    src_bands_db = spectral_energy_at_bands(audio, sr, band_edges)
    ref_bands_db = spectral_energy_at_bands(ref_audio, ref_sr, band_edges)
    match_before = spectral_match_score(src_bands_db, ref_bands_db)

    curve = compute_reference_eq_curve(src_bands_db, ref_bands_db, centers,
                                       max_boost_db=eq_max_boost_db,
                                       max_cut_db=eq_max_cut_db,
                                       blend=eq_match_blend)

    # ── 2. EQ de matching (FIR de fase lineal, ver build_matching_fir) ────
    _report(progress_cb, 20, "Calculando y aplicando EQ de matching (FIR)")
    audio = eq_high_pass(audio, sr, cutoff_hz=hp_cutoff)
    fir_taps = build_matching_fir(curve, sr, precision=eq_q)
    audio = apply_matching_fir(audio, sr, fir_taps)

    post_eq_bands_db = spectral_energy_at_bands(audio, sr, band_edges)
    match_after_eq = spectral_match_score(post_eq_bands_db, ref_bands_db)

    # ── 3. Dinámica: matching multibanda (graves/medios/agudos) por crest   ──
    # ── factor + matching de macro-dinámica (LRA). Ver match_dynamics_bands ──
    # ── y match_lra: ambas funciones SOLO comprimen (nunca expanden), banda ──
    # ── por banda, comparando cada rango de frecuencia contra el mismo      ──
    # ── rango de la referencia (en vez de un único crest factor global).    ──
    dynamics_band_meta = {name: {"applied": False, "gap_db": 0.0} for name, _, _ in DYNAMICS_BANDS}
    lra_meta = {"applied": False, "own_lra": analysis_before.get("lra", 0.0),
               "ref_lra": analysis_reference.get("lra", 0.0)}
    _report(progress_cb, 40, "Igualando dinámica contra la referencia")
    if match_dynamics:
        own_crest = band_crest_factors(audio, sr)
        ref_crest = band_crest_factors(ref_audio, ref_sr)
        audio, dynamics_band_meta = match_dynamics_bands(
            audio, sr, own_crest, ref_crest, margin_db=dynamics_margin_db,
            oversample=ovs)

        cur_lra = measure_lra(audio, sr)
        audio, lra_meta = match_lra(audio, sr, cur_lra, analysis_reference.get("lra", cur_lra),
                                    margin=dynamics_margin_db, oversample=ovs)

    # ── 4. Ancho estéreo: matching banda por banda (graves/medios/agudos)   ──
    # ── de la correlación L/R contra la referencia. Ver match_stereo_bands. ──
    stereo_k_applied = {name: 1.0 for name, _, _ in DYNAMICS_BANDS}
    _report(progress_cb, 60, "Igualando ancho estéreo contra la referencia")
    if match_stereo_width and audio.ndim == 2 and audio.shape[0] == 2 and ref_audio.ndim == 2 and ref_audio.shape[0] == 2:
        ref_band_corr = band_stereo_correlation(ref_audio, ref_sr)
        audio, stereo_k_applied = match_stereo_bands(
            audio, sr, ref_band_corr, blend=stereo_blend)
    width_applied = round(float(np.mean(list(stereo_k_applied.values()))), 3)

    # ── 5. Loudness (LUFS) ─────────────────────────────────────────────────
    _report(progress_cb, 75, "Igualando loudness (LUFS) contra la referencia")
    loudness_gain_db = 0.0
    if match_loudness:
        cur_lufs = measure_lufs_integrated(audio, sr)
        loudness_gain_db = float(np.clip(analysis_reference["lufs"] - cur_lufs, -24.0, 24.0))
        audio = audio * (10.0 ** (loudness_gain_db / 20.0))

    # ── 6. Limitador, techo aproximado al pico de la referencia ───────────
    _report(progress_cb, 85, "Aplicando limitador final")
    ref_peak_db = min(analysis_reference["peak_db"], -0.1)
    ceiling = float(np.clip(10.0 ** (ref_peak_db / 20.0), 0.5, 0.99))
    audio = limiter(audio, sr, ceiling=ceiling, release_ms=limiter_release_ms, lookahead_ms=5.0,
                    oversample=ovs)

    _report(progress_cb, 93, "Analizando resultado final")
    analysis_after = analyze_audio(audio, sr)
    final_bands_db = spectral_energy_at_bands(audio, sr, band_edges)
    match_after = spectral_match_score(final_bands_db, ref_bands_db)

    # ── 7. Reporte de análisis inteligente (resume las 4 dimensiones) ─────
    intelligent_report = reference_intelligent_report(
        match_after, loudness_gain_db, dynamics_band_meta, lra_meta, stereo_k_applied)

    base = os.path.splitext(os.path.basename(input_path))[0]
    suffix = "_preview_refmatch" if preview_seconds else "_refmatch"
    output_path = f"processed/{base}_{uuid.uuid4().hex[:8]}{suffix}.{output_format}"

    audio_out = audio[0] if audio.shape[0] == 1 else audio.T

    _report(progress_cb, 97, "Guardando archivo masterizado")
    effective_bit_depth = _write_master_output(
        audio_out, sr, output_path, output_format, output_bit_depth
    )

    _report(progress_cb, 100, "Mastering por referencia completado")
    return {
        "output_path":       output_path,
        "output_bit_depth":  effective_bit_depth,
        "analysis_before":   analysis_before,
        "analysis_after":    analysis_after,
        "analysis_reference": analysis_reference,
        "mix_advice_before": mix_advice(analysis_before),
        "mix_advice_after":  mix_advice(analysis_after),
        "reference_match": {
            "before":                  match_before,
            "after_eq":                match_after_eq,
            "after":                   match_after,
            "eq_curve_db":             [{"freq_hz": round(f, 1), "gain_db": round(g, 2)} for f, g in curve],
            "loudness_gain_applied_db": round(loudness_gain_db, 2),
            "eq_match_blend":          round(eq_match_blend, 3),
            "oversample":              ovs,
            "oversample_mode":         str(oversample_mode),
            "stereo_width_applied":     round(width_applied, 3),
            "stereo_width_by_band":     stereo_k_applied,
            "dynamics_by_band":         dynamics_band_meta,
            "lra":                      lra_meta,
            "limiter_ceiling":          round(ceiling, 4),
            "intelligent_report":       intelligent_report,
        },
        "chain_meters": {
            "post_limiter": {
                "rms_db":  analysis_after["rms_db"],
                "peak_db": analysis_after["peak_db"],
                "lufs":    analysis_after["lufs"],
                "lra":     analysis_after.get("lra"),
                "stereo_correlation": analysis_after["stereo_correlation"],
            },
        },
    }