"""
ai_assistant.py — Asistente de IA conversacional para mastering (estilo LANDR AI).

Usa la API de Google Gemini para responder preguntas del usuario sobre su
mezcla/master, apoyándose en el análisis técnico ya calculado por mastering.py
(analyze_audio / mix_advice / spectrum) como contexto, más los presets y
targets de loudness disponibles en la app.

No requiere que el usuario suba el audio de nuevo: el frontend le pasa el
último resultado de /analyze (o el `result.metrics` de un job de /master) y
este módulo arma un prompt de sistema con ese contexto para que las
respuestas sean específicas ("tu LUFS está en -8.2, muy caliente para
Spotify...") en vez de genéricas.
"""

from __future__ import annotations

import os
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Carga opcional de un archivo .env (backend/.env) si python-dotenv está instalado.
# Si no está instalado o no hay .env, simplemente seguimos leyendo os.environ tal cual
# (útil si la API key ya viene seteada por el sistema/servicio/Docker).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Acepta GEMINI_API_KEY o GOOGLE_API_KEY (el SDK de Google usa cualquiera de las dos).
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
AI_MODEL = os.environ.get("AI_ASSISTANT_MODEL", "gemini-2.5-flash")
MAX_HISTORY_MESSAGES = 20  # últimos N mensajes de la conversación que se reenvían

_client = None
_client_error: Optional[str] = None


def _get_client():
    """Instancia el cliente de Gemini de forma perezosa (lazy) y cachea errores."""
    global _client, _client_error
    if _client is not None:
        return _client
    if _client_error is not None:
        return None
    if not GEMINI_API_KEY:
        _client_error = (
            "Falta configurar la variable de entorno GEMINI_API_KEY en el backend."
        )
        logger.warning(_client_error)
        return None
    try:
        from google import genai
        _client = genai.Client(api_key=GEMINI_API_KEY)
        return _client
    except Exception as e:  # paquete no instalado, key inválida, etc.
        _client_error = f"No se pudo inicializar el cliente de Gemini: {e}"
        logger.error(_client_error)
        return None


def is_available() -> bool:
    return _get_client() is not None


def get_unavailable_reason() -> str:
    return _client_error or "El asistente de IA no está disponible."


def _fmt(v, unit: str = "", nd: int = 2) -> str:
    if v is None:
        return "sin datos"
    try:
        return f"{round(float(v), nd)}{unit}"
    except (TypeError, ValueError):
        return str(v)


def build_audio_context(analysis: Optional[dict], preset: Optional[str] = None,
                         platform: Optional[str] = None) -> str:
    """Convierte el dict de analyze_audio()/mix_advice() en texto legible para el modelo."""
    if not analysis:
        return "El usuario todavía no subió ningún audio ni corrió un análisis en esta sesión."

    lines = ["Datos técnicos del análisis del track actual del usuario (decenas de métricas ya calculadas):"]

    lines.append("· Loudness / nivel:")
    lines.append(f"    - LUFS integrado: {_fmt(analysis.get('lufs'), ' LUFS')}")
    lines.append(f"    - Pico (sample): {_fmt(analysis.get('peak_db'), ' dBFS')}")
    lines.append(f"    - True peak (inter-sample, 4x oversample): {_fmt(analysis.get('true_peak_db'), ' dBTP')}")
    lines.append(f"    - RMS: {_fmt(analysis.get('rms_db'), ' dBFS')}")
    lines.append(f"    - PLR (true peak - LUFS): {_fmt(analysis.get('plr_db'), ' dB')}")
    st = analysis.get("loudness_short_term") or {}
    if st:
        lines.append(f"    - Loudness de corto plazo (ventanas 3s): máx {_fmt(st.get('max'), ' LUFS')}, "
                      f"mín {_fmt(st.get('min'), ' LUFS')}, p95 {_fmt(st.get('p95'), ' LUFS')}")
    lines.append(f"    - LRA (rango de loudness): {_fmt(analysis.get('lra'), ' LU')}")

    lines.append("· Dinámica:")
    lines.append(f"    - Rango dinámico global (crest factor): {_fmt(analysis.get('dynamic_range_db'), ' dB')}")
    band_dyn = analysis.get("band_dynamics_db") or {}
    if band_dyn:
        lines.append(f"    - Crest factor por banda: graves {_fmt(band_dyn.get('low'), ' dB')}, "
                      f"medios {_fmt(band_dyn.get('mid'), ' dB')}, agudos {_fmt(band_dyn.get('high'), ' dB')}")

    lines.append("· Higiene de señal:")
    lines.append(f"    - DC offset: {_fmt(analysis.get('dc_offset'), '', 5)}")
    lines.append(f"    - Clipping real: {_fmt((analysis.get('clipping_ratio') or 0) * 100, '% de las muestras', 3)}")
    lines.append(f"    - Silencio (<-60dB): {_fmt((analysis.get('silence_ratio') or 0) * 100, '% del track', 2)}")

    lines.append("· Estéreo:")
    lines.append(f"    - Correlación L/R global: {_fmt(analysis.get('stereo_correlation'), '', 3)}")
    band_st = analysis.get("band_stereo_correlation") or {}
    if band_st:
        lines.append(f"    - Correlación L/R por banda: graves {_fmt(band_st.get('low'), '', 3)}, "
                      f"medios {_fmt(band_st.get('mid'), '', 3)}, agudos {_fmt(band_st.get('high'), '', 3)}")
    lines.append(f"    - Compatibilidad mono (pérdida al sumar L+R): {_fmt(analysis.get('mono_compatibility_db'), ' dB')}")

    lines.append("· Timbre / forma espectral:")
    lines.append(f"    - Centroid espectral: {_fmt(analysis.get('spectral_centroid_hz'), ' Hz', 0)}")
    lines.append(f"    - Rolloff (85% energía): {_fmt(analysis.get('spectral_rolloff_hz'), ' Hz', 0)}")
    lines.append(f"    - Flatness espectral (0=tonal, 1=ruidoso): {_fmt(analysis.get('spectral_flatness'), '', 4)}")
    lines.append(f"    - Zero-crossing rate: {_fmt(analysis.get('zero_crossing_rate'), '', 4)}")

    lines.append("· Ritmo:")
    if analysis.get("bpm"):
        lines.append(f"    - Tempo estimado: {_fmt(analysis.get('bpm'), ' BPM', 1)}")
    lines.append(f"    - Densidad de transientes: {_fmt(analysis.get('transient_density'), ' onsets/seg', 2)}")

    spectrum = analysis.get("spectrum") or {}
    if spectrum:
        band_names = {
            "sub_bass": "Sub-bajos (20-80Hz)", "bass": "Bajos (80-250Hz)",
            "low_mid": "Medios-bajos (250-500Hz)", "mid": "Medios (500-2kHz)",
            "upper_mid": "Medios-altos (2-4kHz)", "presence": "Presencia (4-8kHz)",
            "air": "Aire (8-20kHz)",
        }
        lines.append("· Balance espectral (energía relativa en dB por banda):")
        for key, label in band_names.items():
            if key in spectrum:
                lines.append(f"    · {label}: {_fmt(spectrum[key], ' dB')}")

    advice = analysis.get("mix_advice")
    if isinstance(advice, dict):
        issues = advice.get("issues") or []
        tips = advice.get("tips") or []
        score = advice.get("score")
        if score is not None:
            lines.append(f"- Score de calidad del motor de reglas interno: {score}/100")
        if issues:
            lines.append("- Problemas detectados automáticamente:")
            for i in issues:
                lines.append(f"    · {i}")
        if tips:
            lines.append("- Sugerencias automáticas del motor de reglas:")
            for t in tips:
                lines.append(f"    · {t}")

    if preset:
        lines.append(f"- Preset de mastering seleccionado por el usuario: {preset}")
    if platform:
        lines.append(f"- Plataforma/target de loudness elegido: {platform}")

    return "\n".join(lines)


SYSTEM_PROMPT_TEMPLATE = """Sos el Asistente de IA de MASTER, un estudio de mastering de audio online \
(similar en espíritu al asistente de IA de LANDR). Hablás en español rioplatense, con tono \
cercano, profesional y directo, como un ingeniero de mastering con experiencia que está \
mirando la sesión del usuario en tiempo real.

Tenés acceso al análisis técnico real del track que subió el usuario (ver más abajo). \
Usalo SIEMPRE que sea relevante para dar respuestas específicas y accionables, citando los \
números concretos (LUFS, dB, balance espectral, etc.) en vez de consejos genéricos.

Además de responder en texto, PODÉS proponer un cambio concreto y aplicable a la cadena de \
mastering (igual que el asistente de LANDR, que no solo aconseja sino que ajusta el master). \
Para eso completá los campos numéricos/booleanos de parámetros que quieras cambiar; dejá en \
null (sin completar) todos los que no correspondan.

Cuándo SÍ proponer parámetros:
- El usuario pide explícitamente un cambio o efecto ("más brillo", "que suene más fuerte", \
"bajale la compresión", "quiero más pegada en el bajo", "subilo a -9 LUFS para Spotify", etc.).
- Vos mismo detectás en el análisis un problema puntual y accionable y el usuario te pidió consejo \
sobre eso (no lo hagas de prepo en preguntas puramente teóricas).

Cuándo NO proponer parámetros (dejar todo en null):
- Preguntas generales de teoría, flujo de trabajo, o que no dependen del track.
- No hay análisis disponible todavía.
- El usuario está charlando, agradeciendo, o pidiendo una aclaración sin pedir un ajuste.

Reglas para las propuestas:
- Cambiá SOLO los parámetros relevantes al pedido puntual (normalmente 1 a 6 campos), NUNCA \
completes todos los campos del esquema como si fuera un mastering completo desde cero.
- Los campos de umbral y techo (comp_threshold_db, mb_low/mid/high_threshold_db, target_peak_db, \
limiter_ceiling_db) están SIEMPRE en dB relativos a 0dBFS (0 = techo digital, valores negativos \
hacia abajo), igual que cualquier otro parámetro en dB de esta lista. No hay ningún parámetro \
en escala lineal 0-1 en este esquema — todo lo que es amplitud/nivel se expresa en dB.
- Los valores deben estar dentro de los rangos válidos (ver más abajo) y ser coherentes con el \
análisis real del track, no genéricos.
- Si proponés parámetros, completá también "suggestion_summary" con una frase muy corta (5-9 \
palabras) que resuma el cambio, ej: "Más aire arriba de 8kHz" o "Bajar 2dB el makeup del compresor".
- En "reply" explicá en 1-3 oraciones qué le vas a cambiar y por qué, en tono conversacional \
(el usuario va a ver un botón aparte para aplicar los valores, no hace falta que listes cada \
número ahí).

Lineamientos generales:
- Sé conciso: respuestas cortas (2-6 oraciones o una lista breve), esto es un chat, no un ensayo.
- Si el usuario pregunta algo que no depende del análisis (teoría, flujo de trabajo, qué preset \
usar, cómo usar la herramienta), respondé igual con tu conocimiento de mastering/mezcla.
- Si no hay análisis disponible todavía, decilo y sugerí analizar o subir un audio primero, pero \
igual podés responder preguntas generales de mastering.
- No inventes datos del track que no estén en el contexto: si falta algo, decí que no lo tenés.
- No sos un modelo genérico: sos parte de esta app de mastering, mantené el foco en audio, \
mezcla, mastering y el uso de la herramienta.

Rangos válidos de los parámetros de la cadena (para cuando propongas cambios):
{ranges_block}

{audio_context}
"""


def chat(user_message: str, history: Optional[list] = None,
         analysis: Optional[dict] = None, preset: Optional[str] = None,
         platform: Optional[str] = None) -> dict:
    """Envía un mensaje al asistente de IA y devuelve un dict:
    {"reply": str, "suggested_params": dict, "suggestion_summary": Optional[str]}

    `suggested_params` viene vacío ({}) cuando el modelo no propuso ningún ajuste \
    aplicable (p.ej. preguntas teóricas) — el frontend solo debe mostrar el botón \
    de "Aplicar cambios" cuando ese dict tiene contenido.

    `history` es una lista de dicts [{"role": "user"|"assistant", "content": str}, ...]
    con los turnos previos de esta conversación (ya sin el mensaje actual).
    """
    client = _get_client()
    if client is None:
        fallback = build_fallback_response(user_message, analysis)
        return {
            "reply": fallback["reply"],
            "suggested_params": fallback["suggested_params"],
            "suggestion_summary": fallback["suggestion_summary"],
        }

    if not user_message or not user_message.strip():
        raise ValueError("El mensaje está vacío.")

    from google.genai import types
    from pydantic import create_model

    ranges_block = _param_ranges_text()
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        ranges_block=ranges_block,
        audio_context=build_audio_context(analysis, preset, platform),
    )

    # Esquema: "reply" es obligatorio; todos los parámetros de la cadena son
    # opcionales (None por defecto) para que el modelo solo complete los que
    # realmente quiere cambiar, en vez de forzar un mastering completo. Los
    # campos de umbral/techo (DB_EXPOSED_FIELDS) se exponen como '{campo}_db':
    # la IA siempre razona y responde en dB, nunca en el ratio lineal interno.
    field_defs = {
        "reply": (str, ...),
        "suggestion_summary": (Optional[str], None),
    }
    for k in PARAM_RANGES:
        if k in DB_EXPOSED_FIELDS:
            continue
        field_defs[k] = (Optional[float], None)
    for k in DB_EXPOSED_FIELDS:
        field_defs[_db_field_name(k)] = (Optional[float], None)
    for k in BOOL_PARAM_FIELDS:
        field_defs[k] = (Optional[bool], None)
    field_defs["saturation_mode"] = (Optional[str], None)
    ChatResponseSchema = create_model("ChatResponseSchema", **field_defs)

    # Gemini usa roles "user" y "model" (en vez de "assistant").
    contents = []
    for turn in (history or [])[-MAX_HISTORY_MESSAGES:]:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and content:
            gemini_role = "model" if role == "assistant" else "user"
            contents.append(types.Content(role=gemini_role, parts=[types.Part(text=content)]))
    contents.append(types.Content(role="user", parts=[types.Part(text=user_message.strip())]))

    response = client.models.generate_content(
        model=AI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=ChatResponseSchema,
            max_output_tokens=2048,
        ),
    )

    data = None
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, ChatResponseSchema):
        data = parsed.model_dump()
    else:
        raw = (getattr(response, "text", None) or "").strip()
        data = _extract_json_object(raw)
        if data is None:
            try:
                finish_reason = response.candidates[0].finish_reason
                logger.warning(f"Respuesta de chat ilegible de Gemini (finish_reason={finish_reason}).")
            except Exception:
                pass

    if not data:
        fallback = build_fallback_response(user_message, analysis)
        return {
            "reply": fallback["reply"],
            "suggested_params": fallback["suggested_params"],
            "suggestion_summary": fallback["suggestion_summary"],
        }

    reply_text = str(data.get("reply") or "").strip() or (
        "No obtuve respuesta del modelo. Probá reformular la pregunta."
    )

    suggested: dict = {}
    for key, (lo, hi) in PARAM_RANGES.items():
        if key in DB_EXPOSED_FIELDS:
            continue
        v = data.get(key)
        if v is None:
            continue
        try:
            clamped = _clamp(float(v), lo, hi)
        except (TypeError, ValueError):
            continue
        if clamped is not None:
            suggested[key] = round(clamped, 3)
    for key in DB_EXPOSED_FIELDS:
        linear_val = _resolve_db_exposed_param(key, data, default_linear=None)
        if linear_val is not None:
            suggested[key] = round(linear_val, 4)
    for key in BOOL_PARAM_FIELDS:
        v = data.get(key)
        if v is not None:
            suggested[key] = bool(v)
    sat_mode = data.get("saturation_mode")
    if sat_mode in SATURATION_MODES:
        suggested["saturation_mode"] = sat_mode

    return {
        "reply": reply_text,
        "suggested_params": suggested,
        "suggestion_summary": (str(data.get("suggestion_summary") or "").strip() or None),
    }


# ═══════════════════════════════════════════════════════════════════════════
# ── Auto-Mastering: la IA toma las decisiones (estilo LANDR AI) ─────────────
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# ── Auto-Mastering: la IA genera los parámetros de la cadena a mano ─────────
# (ya NO elige entre presets fijos: calcula cada valor en base al análisis)
# ═══════════════════════════════════════════════════════════════════════════

# Rango válido [min, max] para cada parámetro numérico de la cadena de mastering.
# Debe reflejar los mismos límites que valida /master en app.py (Query ge/le),
# para que la IA nunca proponga un valor que el motor de audio vaya a rechazar.
PARAM_RANGES: dict[str, tuple[float, float]] = {
    "input_gain_db": (-24.0, 24.0),
    "target_peak": (0.1, 1.0),
    "target_lufs": (-40.0, 0.0),
    "hp_cutoff": (20.0, 500.0),
    "high_shelf_gain_db": (-12.0, 12.0),
    "eq1_freq": (20.0, 20000.0), "eq1_gain": (-12.0, 12.0), "eq1_q": (0.1, 10.0),
    "eq2_freq": (20.0, 20000.0), "eq2_gain": (-12.0, 12.0), "eq2_q": (0.1, 10.0),
    "eq3_freq": (20.0, 20000.0), "eq3_gain": (-12.0, 12.0), "eq3_q": (0.1, 10.0),
    "eq4_freq": (20.0, 20000.0), "eq4_gain": (-12.0, 12.0), "eq4_q": (0.1, 10.0),
    "comp_threshold": (0.0, 1.0), "comp_ratio": (1.0, 20.0),
    "comp_attack_ms": (0.1, 200.0), "comp_release_ms": (10.0, 1000.0), "comp_makeup_db": (-12.0, 24.0),
    "transient_attack": (-1.0, 1.0), "transient_sustain": (-1.0, 1.0),
    "mb_low_crossover": (20.0, 2000.0), "mb_high_crossover": (500.0, 20000.0),
    "mb_low_threshold": (0.0, 1.0), "mb_low_ratio": (1.0, 20.0), "mb_low_attack_ms": (0.1, 200.0), "mb_low_release_ms": (10.0, 1000.0), "mb_low_makeup_db": (-12.0, 24.0),
    "mb_mid_threshold": (0.0, 1.0), "mb_mid_ratio": (1.0, 20.0), "mb_mid_attack_ms": (0.1, 200.0), "mb_mid_release_ms": (10.0, 1000.0), "mb_mid_makeup_db": (-12.0, 24.0),
    "mb_high_threshold": (0.0, 1.0), "mb_high_ratio": (1.0, 20.0), "mb_high_attack_ms": (0.1, 200.0), "mb_high_release_ms": (10.0, 1000.0), "mb_high_makeup_db": (-12.0, 24.0),
    "saturation_drive": (0.0, 1.0), "saturation_mix": (0.0, 1.0),
    "mid_gain_db": (-12.0, 12.0), "side_gain_db": (-18.0, 18.0), "stereo_width_amount": (0.0, 3.0),
    "enhancer_bass_mono_freq": (40.0, 500.0), "haas_delay_ms": (0.0, 30.0),
    "reverb_size": (0.05, 2.0), "reverb_wet": (0.0, 1.0),
    "limiter_ceiling": (0.5, 1.0), "limiter_release_ms": (1.0, 500.0),
}
BOOL_PARAM_FIELDS = ["use_lufs_normalize", "mb_bypass", "use_stereo_enhancer"]
SATURATION_MODES = ("tape", "tube")

# Campos de umbral/techo del motor que son ratios lineales de amplitud (0-1,
# relativos a 0dBFS) — nomenclatura técnica interna del DSP. Se los exponemos a
# la IA (y al chat) directamente en dB, que es la escala en la que cualquiera
# razona naturalmente sobre audio, en vez de forzarla a pensar en ratios 0-1.
# El código convierte a la escala lineal real del motor antes de aplicar nada;
# la IA nunca ve ni produce el número lineal.
DB_EXPOSED_FIELDS: dict[str, tuple[float, float]] = {
    "target_peak": (-20.0, 0.0),
    "comp_threshold": (-40.0, 0.0),
    "mb_low_threshold": (-40.0, 0.0),
    "mb_mid_threshold": (-40.0, 0.0),
    "mb_high_threshold": (-40.0, 0.0),
    "limiter_ceiling": (-6.0, 0.0),
}


def _db_field_name(key: str) -> str:
    """Nombre del campo tal como lo ve la IA: 'comp_threshold' -> 'comp_threshold_db'."""
    return f"{key}_db"


def _db_to_linear(db_value: float) -> float:
    return 10 ** (db_value / 20.0)


def _linear_to_db(linear_value: float) -> float:
    import math
    return round(20 * math.log10(max(float(linear_value), 1e-6)), 2)


# Aclaraciones de escala/unidad para el resto de parámetros que no son dB reales
# (los de DB_EXPOSED_FIELDS ya no necesitan nota: se exponen directamente en dB).
PARAM_NOTES: dict[str, str] = {
    "saturation_drive": "0–1, cantidad de saturación (no dB). Típico 0.05–0.3.",
    "saturation_mix": "0–1, mezcla dry/wet (no dB). Típico 0.1–0.4.",
    "reverb_wet": "0–1, mezcla dry/wet (no dB). Típico 0–0.1, es mastering, no mezcla.",
    "reverb_size": "0.05–2, tamaño relativo de la reverb (no dB, no segundos).",
    "transient_attack": "-1 a 1, unitless. Negativo = atenúa transitorios, positivo = realza.",
    "transient_sustain": "-1 a 1, unitless. Negativo = atenúa sustain, positivo = realza.",
    "stereo_width_amount": "multiplicador de ancho estéreo, 1.0 = ancho original (no dB, no %).",
    "comp_ratio": "ratio de compresión X:1 (ej. 2.5 = 2.5:1), no dB.",
    "mb_low_ratio": "ratio de compresión X:1, no dB.",
    "mb_mid_ratio": "ratio de compresión X:1, no dB.",
    "mb_high_ratio": "ratio de compresión X:1, no dB.",
}


def _param_ranges_text() -> str:
    """Arma el bloque de rangos válidos para el prompt. Los campos de umbral/techo \
    (DB_EXPOSED_FIELDS) se listan ya convertidos a dB, con su nombre '_db' — la IA \
    nunca ve el ratio lineal interno del motor."""
    lines = []
    for key, (lo, hi) in PARAM_RANGES.items():
        if key in DB_EXPOSED_FIELDS:
            continue  # se listan más abajo, en dB
        note = PARAM_NOTES.get(key)
        if note:
            lines.append(f"- {key}: [{lo}, {hi}] — {note}")
        elif key.endswith("_db"):
            lines.append(f"- {key}: [{lo}, {hi}] dB")
        elif key.endswith("_ms"):
            lines.append(f"- {key}: [{lo}, {hi}] ms")
        elif key.endswith(("freq", "crossover", "cutoff")) or key.endswith("_hz"):
            lines.append(f"- {key}: [{lo}, {hi}] Hz")
        else:
            lines.append(f"- {key}: [{lo}, {hi}]")
    for key, (db_lo, db_hi) in DB_EXPOSED_FIELDS.items():
        lines.append(f"- {_db_field_name(key)}: [{db_lo}, {db_hi}] dB, relativo a 0dBFS "
                      f"(el motor lo convierte solo a ratio interno, no calcules vos la conversión)")
    return "\n".join(lines)


def _resolve_db_exposed_param(key: str, data: dict, default_linear: Optional[float] = None):
    """Lee el campo '{key}_db' de la respuesta de la IA, lo clampea en dB y lo \
    convierte a la escala lineal real del motor (PARAM_RANGES[key]). Si no vino \
    en la respuesta, devuelve `default_linear` (ya en escala lineal) sin tocar."""
    db_lo, db_hi = DB_EXPOSED_FIELDS[key]
    lo, hi = PARAM_RANGES[key]
    raw = data.get(_db_field_name(key))
    if raw is None:
        return default_linear
    try:
        db_val = _clamp(float(raw), db_lo, db_hi)
    except (TypeError, ValueError):
        return default_linear
    if db_val is None:
        return default_linear
    linear_val = _clamp(_db_to_linear(db_val), lo, hi)
    return linear_val if linear_val is not None else default_linear


# Valores neutros de partida: no representan ningún género en particular, son
# sólo el punto de referencia que se usa si la IA no está disponible y hay
# que recurrir a la heurística de respaldo (ver _fallback_custom_params).
_NEUTRAL_PARAMS: dict = {
    "input_gain_db": 0.0, "target_peak": 0.95, "use_lufs_normalize": False, "target_lufs": -12.0,
    "hp_cutoff": 35.0, "high_shelf_gain_db": 1.5,
    "eq1_freq": 100.0, "eq1_gain": 0.0, "eq1_q": 1.0,
    "eq2_freq": 400.0, "eq2_gain": 0.0, "eq2_q": 1.1,
    "eq3_freq": 2500.0, "eq3_gain": 0.0, "eq3_q": 1.0,
    "eq4_freq": 9000.0, "eq4_gain": 0.5, "eq4_q": 0.9,
    "comp_threshold": 0.55, "comp_ratio": 2.2, "comp_attack_ms": 12.0, "comp_release_ms": 120.0, "comp_makeup_db": 1.0,
    "transient_attack": 0.0, "transient_sustain": 0.0,
    "mb_bypass": False,
    "mb_low_crossover": 150.0, "mb_high_crossover": 4000.0,
    "mb_low_threshold": 0.60, "mb_low_ratio": 2.0, "mb_low_attack_ms": 20.0, "mb_low_release_ms": 150.0, "mb_low_makeup_db": 0.3,
    "mb_mid_threshold": 0.62, "mb_mid_ratio": 1.8, "mb_mid_attack_ms": 15.0, "mb_mid_release_ms": 120.0, "mb_mid_makeup_db": 0.3,
    "mb_high_threshold": 0.65, "mb_high_ratio": 1.6, "mb_high_attack_ms": 8.0, "mb_high_release_ms": 90.0, "mb_high_makeup_db": 0.2,
    "saturation_drive": 0.1, "saturation_mode": "tape", "saturation_mix": 0.2,
    "mid_gain_db": 0.0, "side_gain_db": 0.0, "stereo_width_amount": 1.05,
    "use_stereo_enhancer": False, "enhancer_bass_mono_freq": 120.0, "haas_delay_ms": 0.0,
    "reverb_size": 0.2, "reverb_wet": 0.02,
    "limiter_ceiling": 0.96, "limiter_release_ms": 55.0,
}

AUTO_MASTER_SYSTEM_PROMPT = """Sos Laia, la ingeniera de mastering de IA integrada en la app MASTER. \
Tu trabajo es decidir de forma 100% autónoma CADA parámetro de la cadena de mastering para el \
track del usuario, con el mismo criterio, cuidado y "consciencia" que aplicaría una ingeniera de \
mastering de estudio de primer nivel escuchando y midiendo el track antes de tocar un solo knob. \
El usuario no va a tocar nada manualmente: confía por completo en tu criterio profesional. \
Trabajás en cuatro fases, en este orden, y tu reasoning final debe reflejar ese razonamiento:

FASE 1 — ANÁLISIS: ya tenés decenas de métricas reales del track (ver más abajo): loudness \
integrado y de corto plazo, true peak, PLR, LRA, dinámica global y por banda (graves/medios/agudos), \
balance espectral de 7 bandas + centroid/rolloff/flatness, correlación estéreo global y por banda, \
compatibilidad mono, DC offset, clipping, silencio, BPM y densidad de transientes. Leelas todas \
antes de decidir nada: son tu diagnóstico, no un formulario a ignorar.

FASE 2 — DECISIÓN INTELIGENTE: a partir de ese diagnóstico, identificá los 2-4 problemas o \
características más importantes de ESTE track puntual (p.ej. "muy comprimido y con exceso de \
graves", o "dinámico pero apagado en agudos", o "buena dinámica, sólo necesita loudness") y \
definí una estrategia de mastering coherente con eso: qué tan agresiva debe ser la compresión, \
si conviene EQ correctiva o de color, cuánto loudness final tiene sentido para el carácter del \
track (no todo tiene que llegar a -6 LUFS), y si el estéreo/saturación necesitan ajuste. \
Dos tracks distintos con métricas distintas DEBEN terminar con parámetros distintos — nunca \
apliques siempre la misma combinación "por defecto".

FASE 3 — CONSTRUCCIÓN DE LA CADENA DSP: traducí la estrategia de la Fase 2 a valores concretos \
de cada etapa de la cadena (en este orden de señal: input gain → high-pass → EQ 4 bandas + high \
shelf → transient shaper → compresor de banda ancha → multibanda → saturación → mid/side/estéreo \
→ reverb → limiter). Guía de criterio (no son reglas rígidas, usalas con sentido común de \
ingeniería, cruzando SIEMPRE con los números reales del análisis):
- Dinámica: si dynamic_range_db o el crest factor por banda ya son bajos (muy comprimido), usá \
compresión y limiter más suaves para no sobre-comprimir; si son altos, podés compensar con más \
ratio/threshold más bajo. LRA muy chico también es señal de sobre-compresión previa.
- Balance espectral: corregí excesos o faltantes de energía en sub-bajos/bajos/medios/presencia/aire \
con hp_cutoff, high_shelf_gain_db y las 4 bandas de EQ paramétrico (eqN_freq/gain/q). El centroid y \
rolloff espectral te dicen si el track suena "oscuro" o "brillante" en términos objetivos.
- Loudness objetivo y ceiling del limiter: dependen de la plataforma elegida (si elegís una) y de \
cuánto necesita "calentarse" el track según su LUFS actual y su PLR (un PLR alto = mucho margen \
para subir loudness sin destruir la dinámica; un PLR bajo = ya está caliente, sé conservador).
- True peak / clipping: si true_peak_db o clipping_ratio ya muestran problemas, usá limiter_ceiling_db \
más conservador (≤ -0.5 dB aprox.) y no agregues makeup gain innecesario.
- Estéreo: si mono_compatibility_db es muy negativo o la correlación por banda en graves es baja, \
NO ensanches más el estéreo (dejá stereo_width_amount cerca de 1.0) y considerá enhancer_bass_mono_freq \
más alto para mantener los graves centrados. Los parámetros multibanda (mb_*) y mid/side son \
correcciones finas; no los actives agresivamente salvo que el análisis lo justifique.
- Densidad de transientes / BPM: material muy percusivo (transient_density alta, trap/metal) tolera \
attack del compresor más lento y más transient shaping; material sostenido (baladas, ambient) pide \
attack más rápido y compresión más suave para no aplastar el groove.
- Sé conservador con saturación, reverb y haas: son color, no arreglos — nunca la solución a un \
problema técnico real.

FASE 4 — OPTIMIZACIÓN: después de tu decisión, el sistema renderiza un preview real con estos \
parámetros y vuelve a medir LUFS y true peak logrados. Si no coinciden con el objetivo, se hace \
una corrección automática de gain-staging (makeup del compresor / input gain) sin tocar tu \
diseño tonal ni dinámico — no necesitás simular esto vos, sólo saber que existe: por eso es más \
importante que definas bien EQ, dinámica y estéreo que perseguir un LUFS exacto de memoria.

Rangos válidos (el motor de audio rechaza cualquier valor fuera de estos límites, así que \
mantenete siempre dentro de ellos):
{ranges_block}

saturation_mode sólo puede ser "tape" o "tube". Los campos use_lufs_normalize, mb_bypass y \
use_stereo_enhancer son booleanos (true/false).

Respondé ÚNICAMENTE con un objeto JSON, sin texto adicional, sin markdown, con TODOS los campos \
numéricos/booleanos de la cadena (los mismos que se listan en los rangos de arriba, más \
saturation_mode), y además:
{{
  "platform": "<una de las plataformas listadas abajo, EXACTAMENTE como está escrita, o null>",
  "reasoning": "<4-6 oraciones en español rioplatense explicando el diagnóstico (Fase 1-2) y las \
decisiones más importantes de la cadena (Fase 3) que tomaste, citando los números concretos del \
análisis (LUFS, true peak, dinámica por banda, balance espectral, estéreo, etc.) y por qué \
elegiste esos valores puntuales de compresor/EQ/limiter>"
}}

Plataformas / targets de loudness disponibles: {platforms_block}

{audio_context}
"""


def _clamp(value, lo, hi):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return max(lo, min(hi, value))


def _parse_instruction_params(user_message: str) -> dict:
    """Traduce instrucciones en lenguaje natural a parámetros de la cadena DSP.

    Ejemplos soportados:
    - 'dame 2 db menos en 4k q 1.2' -> EQ banda 3 con gain -2 dB, freq 4k, Q 1.2
    - 'subí 1.5 dB el aire' -> high_shelf_gain_db +1.5
    - 'más compresión' -> comp_ratio/comp_threshold más agresivos
    - 'más reverb' -> reverb_wet +0.04
    """
    message = (user_message or "").strip().lower()
    if not message:
        return {}

    params: dict = {}
    summary = None
    reply = None

    def add_param(key, value):
        if value is None:
            return
        if isinstance(value, bool):
            params[key] = value
        elif isinstance(value, str):
            params[key] = value
        else:
            params[key] = round(float(value), 4)

    # --- Gain / level parsing ---
    gain_match = re.search(r'([+-]?\d+(?:[.,]\d+)?)\s*(?:db|dB|decibeles|decibel)', message)
    gain_db = None
    if gain_match:
        gain_db = float(gain_match.group(1).replace(",", "."))
        if re.search(r'\b(menos|bajar|bajá|bajale|restar|cut|down)\b', message):
            gain_db = -abs(gain_db)
        elif re.search(r'\b(subir|subi|subile|aumentar|agregar|boost|up|más)\b', message):
            gain_db = abs(gain_db)

    # --- Frequency parsing ---
    freq_hz = None
    freq_match = re.search(r'(\d+(?:[.,]\d+)?)\s*(k|khz|hz)', message)
    if freq_match:
        val = float(freq_match.group(1).replace(",", "."))
        unit = freq_match.group(2).lower()
        if unit in {"k", "khz"}:
            freq_hz = val * 1000.0
        else:
            freq_hz = val

    # --- Q parsing ---
    q_value = None
    q_match = re.search(r'(?:q|q=|q:|q\s*)(\d+(?:[.,]\d+)?)', message)
    if q_match:
        q_value = float(q_match.group(1).replace(",", "."))

    # --- EQ instruction: 'en 4k', 'en 8k', '2 db menos en 4k q 1.2' ---
    if gain_db is not None and ("eq" in message or freq_hz is not None or "4k" in message or "8k" in message or "khz" in message):
        if freq_hz is None and gain_db is not None:
            freq_hz = 4000.0 if "4k" in message or "4 khz" in message else 8000.0 if "8k" in message or "8 khz" in message else None
        if freq_hz is not None:
            if freq_hz <= 180.0:
                band_key = "eq1"
            elif freq_hz <= 1000.0:
                band_key = "eq2"
            elif freq_hz <= 3000.0:
                band_key = "eq3"
            else:
                band_key = "eq4"
            add_param(f"{band_key}_freq", freq_hz)
            add_param(f"{band_key}_gain", gain_db)
            if q_value is not None:
                add_param(f"{band_key}_q", q_value)
            summary = "Ajuste de EQ paramétrico"
            reply = f"Ajusté la banda de EQ en {int(freq_hz/1000)} kHz con {gain_db:+.1f} dB."

    # --- Shelf / air / brilliance ---
    if gain_db is not None and ("aire" in message or "brillo" in message or "agudos" in message or "high shelf" in message):
        add_param("high_shelf_gain_db", gain_db)
        if summary is None:
            summary = "Ajuste de aire en agudos"
            reply = f"Subí el shelf de agudos en {gain_db:+.1f} dB."

    # --- Compression controls ---
    if "compres" in message or "compress" in message or "compression" in message:
        if gain_db is not None and ("threshold" in message or "umbral" in message or "thresh" in message):
            add_param("comp_threshold", _db_to_linear(gain_db))
            summary = summary or "Ajuste de compresión"
            reply = reply or f"Cambié el threshold del compresor a {gain_db:+.1f} dB."
        elif "ratio" in message or "ratio" in message:
            ratio_match = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:x|:1|ratio)', message)
            if ratio_match:
                add_param("comp_ratio", float(ratio_match.group(1).replace(",", ".")))
                summary = summary or "Ajuste de ratio"
        elif "attack" in message or "ataque" in message:
            ms_match = re.search(r'(\d+(?:[.,]\d+)?)\s*(ms|miliseg)', message)
            if ms_match:
                add_param("comp_attack_ms", float(ms_match.group(1).replace(",", ".")))
        elif "release" in message or "release" in message or "soltar" in message:
            ms_match = re.search(r'(\d+(?:[.,]\d+)?)\s*(ms|miliseg)', message)
            if ms_match:
                add_param("comp_release_ms", float(ms_match.group(1).replace(",", ".")))

    # --- Saturation ---
    if "satur" in message or "drive" in message:
        if gain_db is not None:
            add_param("saturation_drive", max(0.0, min(1.0, abs(gain_db) / 12.0)))
            summary = summary or "Ajuste de saturación"

    # --- Stereo / width ---
    if "ancho" in message or "estéreo" in message or "width" in message:
        width_match = re.search(r'(\d+(?:[.,]\d+)?)\s*(x|times)', message)
        if width_match:
            add_param("stereo_width_amount", float(width_match.group(1).replace(",", ".")))
        elif gain_db is not None:
            add_param("stereo_width_amount", max(0.0, min(3.0, 1.0 + gain_db / 6.0)))

    # --- Reverb ---
    if "reverb" in message or "verb" in message:
        if gain_db is not None:
            add_param("reverb_wet", max(0.0, min(1.0, abs(gain_db) / 12.0)))
        else:
            add_param("reverb_wet", 0.08)

    # --- Limiter / ceiling / loudness ---
    if "limiter" in message or "techo" in message or "pico" in message:
        if gain_db is not None:
            add_param("limiter_ceiling", _db_to_linear(gain_db))
            summary = summary or "Ajuste de limiter"
    if "loud" in message or "lufs" in message or "fuerte" in message or "más fuerte" in message:
        if gain_db is not None:
            add_param("comp_makeup_db", gain_db)
            summary = summary or "Ajuste de loudness"

    # --- Parámetros avanzados / modos de cadena ---
    if "linear phase" in message or "phase linear" in message or "linear_phase" in message:
        add_param("eq_mode", "linear_phase")
    if "iir" in message:
        add_param("eq_mode", "iir")
    if "glue" in message and ("bypass" in message or "desactiv" in message or "apag" in message):
        add_param("glue_bypass", True)
    if "glue" in message and ("activar" in message or "encend" in message or "on" in message):
        add_param("glue_bypass", False)
    if "multibanda" in message and ("bypass" in message or "desactiv" in message or "apag" in message):
        add_param("mb_bypass", True)
    if "multibanda" in message and ("activar" in message or "encend" in message or "on" in message):
        add_param("mb_bypass", False)
    if "estéreo" in message and ("enhancer" in message or "estereo enhancer" in message):
        add_param("use_stereo_enhancer", True)
    if "enhancer" in message and ("bypass" in message or "desactiv" in message or "apag" in message):
        add_param("use_stereo_enhancer", False)
    if "link" in message and "stereo" in message and ("activar" in message or "on" in message):
        add_param("comp_stereo_link", True)
    if "link" in message and "stereo" in message and ("desactiv" in message or "apag" in message):
        add_param("comp_stereo_link", False)
    if "oversample" in message or "oversampling" in message:
        if "bajo" in message or "draft" in message or "fast" in message or "rapido" in message:
            add_param("oversample_mode", "fast")
        elif "alta" in message or "quality" in message or "calidad" in message or "ultra" in message:
            add_param("oversample_mode", "quality")
        else:
            add_param("oversample_mode", "quality")

    return params if params else {}


def build_fallback_response(user_message: str, analysis: Optional[dict]) -> dict:
    """Respuesta de respaldo útil cuando la IA externa no está disponible o no
    devuelve cambios accionables. Genera sugerencias simples pero realistas basadas
    en el análisis y en palabras claves del mensaje del usuario."""
    message = (user_message or "").strip().lower()
    parsed_params = _parse_instruction_params(user_message)
    if parsed_params:
        return {
            "reply": f"Aplicaré ese ajuste directamente en la cadena DSP: {', '.join(parsed_params.keys())}.",
            "suggested_params": parsed_params,
            "suggestion_summary": "Ajuste DSP guiado por texto",
        }

    a = analysis or {}
    spectrum = a.get("spectrum") or {}
    advice = a.get("mix_advice") or {}
    issues = [str(i).lower() for i in (advice.get("issues") or [])]
    tips = [str(t).lower() for t in (advice.get("tips") or [])]
    lufs = a.get("lufs")
    peak = a.get("peak_db")
    true_peak = a.get("true_peak_db")

    suggested: dict = {}
    summary = None
    reply = "Te propongo un ajuste conservador según el análisis del track."

    def pick(value, key):
        if value is None:
            return
        suggested[key] = round(float(value), 3)

    wants_brighter = any(k in message for k in ["brillo", "aire", "agudos", "bright", "shine", "más brillo"])
    wants_louder = any(k in message for k in ["más fuerte", "louder", "subilo", "subir", "loudness", "más loud"])
    wants_less_comp = any(k in message for k in ["menos comp", "menos compresión", "suave", "más natural", "relajá"])
    wants_more_warmth = any(k in message for k in ["calido", "warm", "grave", "graves", "bajo"])
    wants_less_clipping = any(k in message for k in ["clipping", "pico", "techo", "limitar", "limiter"])

    if wants_brighter or any("altas frecuencias muy bajas" in i for i in issues) or any("high shelf" in t for t in tips):
        boost = 2.0 if (spectrum.get("air") is not None and float(spectrum.get("air", -999)) < -24) else 1.5
        pick(boost, "high_shelf_gain_db")
        pick(min(2.0, max(0.8, boost - 0.4)), "eq4_gain")
        summary = "Más aire y brillo en agudos"
        reply = "Voy a sumar un poco de aire en los agudos para que el track se vea más abierto y brillante."

    if wants_louder or (isinstance(lufs, (int, float)) and lufs < -18):
        if isinstance(lufs, (int, float)) and lufs < -20:
            pick(-12.0, "target_lufs")
            pick(1.5, "comp_makeup_db")
            summary = summary or "Subir loudness con más control"
            reply = "El track está bastante abajo en loudness, así que priorizo un lift de nivel con compresión y limiter más controlados."
        else:
            pick(-14.0, "target_lufs")
            pick(0.8, "comp_makeup_db")
            summary = summary or "Subir loudness sin perder cuerpo"
            reply = "Voy a empujar un poco el nivel general para que quede más presente sin perder demasiada dinámica."

    if wants_less_comp or any("muy comprimido" in i or "muy comprimido" in t for i, t in zip(issues, tips)):
        pick(0.65, "comp_threshold")
        pick(2.0, "comp_ratio")
        summary = summary or "Compresión más natural"
        reply = "Voy a aflojar un poco la compresión para que el track se sienta menos aplastado."

    if wants_more_warmth or (spectrum.get("bass") is not None and float(spectrum.get("bass", -999)) < -18):
        pick(45.0, "hp_cutoff")
        pick(1.2, "eq1_gain")
        summary = summary or "Más cuerpo en bajos"
        reply = "Voy a reforzar la zona de bajos y limpiar un poco el extremo grave para que suene más sólido."

    if wants_less_clipping or ((isinstance(true_peak, (int, float)) and true_peak > -0.5) or (isinstance(peak, (int, float)) and peak > -0.5)):
        pick(0.94, "limiter_ceiling")
        pick(50.0, "limiter_release_ms")
        summary = summary or "Más margen de pico"
        reply = "Voy a bajar un poco el techo del limiter para reducir riesgo de clipping y proteger la salida."

    if not suggested:
        if isinstance(lufs, (int, float)) and lufs < -18:
            pick(-14.0, "target_lufs")
            pick(0.8, "comp_makeup_db")
            summary = "Subir loudness con más control"
        elif any("pico" in i for i in issues):
            pick(0.94, "limiter_ceiling")
            summary = "Reducir riesgo de clipping"
        elif any("muy bajo" in i for i in issues):
            pick(1.0, "high_shelf_gain_db")
            summary = "Aumentar claridad general"

    return {
        "reply": reply,
        "suggested_params": suggested,
        "suggestion_summary": summary,
    }


def _fallback_custom_params(analysis: Optional[dict]) -> dict:
    """Heurística de respaldo (sin IA): parte de valores neutros y los ajusta \
    a mano según el análisis real del track, en vez de aplicar un preset fijo."""
    p = dict(_NEUTRAL_PARAMS)
    a = analysis or {}
    lufs = a.get("lufs")
    dyn = a.get("dynamic_range_db")
    spectrum = a.get("spectrum") or {}

    # Dinámica: si ya viene muy comprimido, aflojamos; si viene muy dinámico, apretamos un poco.
    if isinstance(dyn, (int, float)):
        if dyn < 6:
            p["comp_ratio"], p["comp_threshold"] = 1.6, 0.70
            p["limiter_release_ms"] = 80.0
        elif dyn > 16:
            p["comp_ratio"], p["comp_threshold"] = 3.0, 0.45
            p["limiter_release_ms"] = 40.0

    # Loudness actual: cuanto más bajo esté, más margen de makeup/limiter le damos.
    if isinstance(lufs, (int, float)):
        if lufs < -20:
            p["comp_makeup_db"] = 2.5
            p["limiter_ceiling"] = 0.97
        elif lufs > -9:
            p["comp_makeup_db"] = 0.0
            p["limiter_ceiling"] = 0.95

    # Balance espectral: corregimos bandas que se salen de un rango razonable.
    def band(key):
        v = spectrum.get(key)
        return v if isinstance(v, (int, float)) else None

    sub = band("sub_bass"); bass = band("bass"); air = band("air"); presence = band("presence")
    if sub is not None and sub > -6:
        p["hp_cutoff"] = 45.0
    if bass is not None and bass < -18:
        p["eq1_gain"] = 2.0
    if presence is not None and presence < -20:
        p["eq3_gain"] = 1.5
    if air is not None and air < -24:
        p["high_shelf_gain_db"] = 3.0
        p["eq4_gain"] = 1.5

    return p


def _resolve_target_lufs(result: dict) -> Optional[float]:
    """Determina el LUFS objetivo de la Fase 4 (Optimización): el de la plataforma \
    elegida si hay una, si no el target_lufs que la propia IA/heurística calculó."""
    from mastering import PLATFORM_LOUDNESS_TARGETS

    platform = result.get("platform")
    if platform and platform in PLATFORM_LOUDNESS_TARGETS:
        return float(PLATFORM_LOUDNESS_TARGETS[platform]["lufs"])
    target = result.get("target_lufs")
    try:
        return float(target)
    except (TypeError, ValueError):
        return None


def _optimize_gain_staging(result: dict, audio, sr, max_iters: int = 2, tolerance_db: float = 0.4) -> list:
    """FASE 4 — Optimización: renderiza un preview real (~25s, del medio del track) con \
    los parámetros que decidió la IA (o la heurística de respaldo), mide el LUFS \
    realmente logrado por la cadena completa y corrige el makeup gain del compresor de \
    banda ancha si hace falta, iterando hasta converger. No toca EQ, dinámica multibanda \
    ni estéreo — sólo hace el gain-staging fino que una ingeniera de mastering haría de \
    oído después de escuchar el primer render. Se salta silenciosamente si no hay audio \
    disponible o no hay un LUFS objetivo definido (nada que optimizar)."""
    if audio is None or sr is None:
        return []
    target_lufs = _resolve_target_lufs(result)
    if target_lufs is None:
        return []

    from mastering import apply_mastering_chain, _crop_preview

    chain_keys = set(PARAM_RANGES.keys()) | set(BOOL_PARAM_FIELDS) | {"saturation_mode"}
    try:
        preview = _crop_preview(audio, sr, 25.0)
    except Exception as e:
        logger.warning(f"Optimización (Fase 4) abortada, no se pudo recortar preview: {e}")
        return []
    if preview.shape[-1] < sr * 2:
        return []  # track demasiado corto para un preview útil

    notes = []
    for i in range(max_iters):
        chain_params = {k: v for k, v in result.items() if k in chain_keys}
        try:
            _, meters = apply_mastering_chain(preview, sr, oversample_mode="fast", **chain_params)
        except Exception as e:
            logger.warning(f"Optimización (Fase 4) abortada, no se pudo renderizar preview: {e}")
            break
        achieved = meters.get("post_limiter", {}).get("lufs")
        if achieved is None:
            break
        delta = target_lufs - achieved
        if abs(delta) <= tolerance_db:
            notes.append(
                f"Optimización: preview verificado en {achieved:.1f} LUFS vs. objetivo "
                f"{target_lufs:.1f} LUFS (dentro de tolerancia, sin corrección adicional)."
            )
            break
        old_makeup = float(result.get("comp_makeup_db", 0.0))
        new_makeup = _clamp(old_makeup + delta, *PARAM_RANGES["comp_makeup_db"])
        if new_makeup is None:
            break
        result["comp_makeup_db"] = round(new_makeup, 3)
        notes.append(
            f"Optimización #{i + 1}: el preview dio {achieved:.1f} LUFS vs. objetivo "
            f"{target_lufs:.1f} LUFS → se corrigió el makeup gain del compresor en {delta:+.1f} dB."
        )
        if abs(new_makeup - old_makeup) < 0.05:
            break  # tocó el límite del rango, no tiene sentido seguir iterando
    return notes


def _apply_optimization(result: dict, audio, sr) -> dict:
    """Corre la Fase 4 y, si hubo correcciones, las suma al reasoning que ya trae `result`."""
    try:
        notes = _optimize_gain_staging(result, audio, sr)
    except Exception as e:
        logger.warning(f"Optimización (Fase 4) falló, se devuelve la decisión sin ajustar: {e}")
        notes = []
    if notes:
        base = (result.get("reasoning") or "").rstrip()
        if base and not base.endswith((".", "!", "?")):
            base += "."
        result["reasoning"] = (base + " " + " ".join(notes)).strip()
    return result


def decide_mastering(analysis: Optional[dict], platform_options: list,
                      audio=None, sr: Optional[int] = None) -> dict:
    """Le pide al modelo que calcule, a mano, todos los parámetros de la cadena \
    de mastering (compresor, EQ de 4 bandas, multibanda, estéreo, limiter, etc.) \
    en base al análisis del track — NO elige entre presets predefinidos.

    `platform_options` es una lista de claves de plataforma válidas. Si se pasan \
    `audio`/`sr` (el mismo array ya cargado que se usó para `analyze_audio`), se \
    corre además la Fase 4 (Optimización): un preview real se renderiza con los \
    parámetros decididos y, si el LUFS logrado no coincide con el objetivo, se \
    corrige el makeup gain e itera hasta converger.

    Devuelve siempre un dict con todos los parámetros validados y clampeados \
    a rango, más 'platform' y 'reasoning', aunque la IA falle o no esté \
    disponible (usa una heurística de respaldo que también calcula los \
    valores a partir del análisis, no de un preset).
    """
    if not analysis:
        raise ValueError(
            "Se necesita analizar el track antes de poder decidir el mastering automático."
        )

    client = _get_client()
    if client is None:
        logger.warning(f"Auto-mastering sin IA disponible ({get_unavailable_reason()}), usando heurística.")
        params = _fallback_custom_params(analysis)
        result = {
            **params, "platform": (platform_options[0] if platform_options else None),
            "reasoning": (
                "No se pudo consultar a la IA, así que se calcularon los parámetros con una "
                "heurística de respaldo en base al rango dinámico, loudness y balance espectral del track."
            ),
        }
        return _apply_optimization(result, audio, sr)

    from google.genai import types
    from pydantic import BaseModel, create_model

    # Esquema estricto generado dinámicamente a partir de PARAM_RANGES, para
    # pedirle a Gemini generación "constrained" con todos los campos de la
    # cadena de mastering (mucho más confiable que solo pedir JSON en el prompt).
    # Los campos de umbral/techo se exponen como '{campo}_db' (ver DB_EXPOSED_FIELDS):
    # la IA siempre calcula y devuelve estos valores en dB, nunca en el ratio lineal
    # interno del motor — el código hace la conversión antes de aplicar nada.
    field_defs = {k: (float, _NEUTRAL_PARAMS.get(k, 0.0)) for k in PARAM_RANGES if k not in DB_EXPOSED_FIELDS}
    for k in DB_EXPOSED_FIELDS:
        field_defs[_db_field_name(k)] = (float, _linear_to_db(_NEUTRAL_PARAMS.get(k, 0.5)))
    for k in BOOL_PARAM_FIELDS:
        field_defs[k] = (bool, _NEUTRAL_PARAMS.get(k, False))
    field_defs["saturation_mode"] = (str, "tape")
    field_defs["platform"] = (Optional[str], None)
    field_defs["reasoning"] = (str, "")
    MasteringParamsSchema = create_model("MasteringParamsSchema", **field_defs)

    ranges_block = _param_ranges_text()
    platforms_block = ", ".join(platform_options) if platform_options else "(ninguna)"
    system_prompt = AUTO_MASTER_SYSTEM_PROMPT.format(
        ranges_block=ranges_block,
        platforms_block=platforms_block,
        audio_context=build_audio_context(analysis),
    )

    data = None
    try:
        response = client.models.generate_content(
            model=AI_MODEL,
            contents=[types.Content(
                role="user",
                parts=[types.Part(text="Calculá los parámetros de mastering para este track y devolvé solo el JSON.")],
            )],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=MasteringParamsSchema,
                max_output_tokens=2048,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, MasteringParamsSchema):
            data = parsed.model_dump()
        else:
            raw = (getattr(response, "text", None) or "").strip()
            data = _extract_json_object(raw)
            if data is None:
                finish_reason = None
                try:
                    finish_reason = response.candidates[0].finish_reason
                except Exception:
                    pass
                logger.error(
                    f"JSON de auto-mastering ilegible incluso con response_schema "
                    f"(finish_reason={finish_reason}). Raw: {raw[:400]!r}"
                )
    except Exception as e:
        logger.error(f"No se pudo obtener/parsear la decisión de mastering de la IA: {e}")

    if not data:
        params = _fallback_custom_params(analysis)
        result = {
            **params, "platform": (platform_options[0] if platform_options else None),
            "reasoning": (
                "La IA no devolvió una respuesta válida, así que se calcularon los parámetros "
                "con una heurística de respaldo en base al análisis del track."
            ),
        }
        return _apply_optimization(result, audio, sr)

    result = {}
    for key, (lo, hi) in PARAM_RANGES.items():
        if key in DB_EXPOSED_FIELDS:
            continue
        try:
            clamped = _clamp(float(data.get(key, _NEUTRAL_PARAMS.get(key))), lo, hi)
        except (TypeError, ValueError):
            clamped = None
        result[key] = clamped if clamped is not None else _NEUTRAL_PARAMS.get(key)
        result[key] = round(result[key], 3)

    for key in DB_EXPOSED_FIELDS:
        linear_val = _resolve_db_exposed_param(key, data, default_linear=_NEUTRAL_PARAMS.get(key))
        result[key] = round(linear_val if linear_val is not None else _NEUTRAL_PARAMS.get(key), 4)

    for key in BOOL_PARAM_FIELDS:
        result[key] = bool(data.get(key, _NEUTRAL_PARAMS.get(key, False)))

    sat_mode = data.get("saturation_mode")
    result["saturation_mode"] = sat_mode if sat_mode in SATURATION_MODES else "tape"

    platform = data.get("platform")
    result["platform"] = platform if platform in (platform_options or []) else None

    result["reasoning"] = str(data.get("reasoning") or "").strip() or (
        "El asistente calculó estos parámetros según el análisis técnico del track."
    )

    return _apply_optimization(result, audio, sr)


def _extract_json_object(raw: str) -> Optional[dict]:
    """Intenta rescatar un dict JSON de una respuesta de texto imperfecta:
    quita fences de markdown, recorta al primer '{'...último '}', y prueba
    arreglos comunes (comas colgantes, comillas simples) antes de rendirse.
    """
    import json
    import re

    if not raw:
        return None
    text = raw.strip()
    # Sacar fences tipo ```json ... ``` o ``` ... ```
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()

    candidates = [text]
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            pass
        # Arreglo básico: comas colgantes antes de } o ]
        fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(fixed)
        except Exception:
            continue
    return None


