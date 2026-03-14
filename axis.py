# =========================================================================
#  A.X.I.S. — Autonomous eXtended Intelligence System
#  Asistente de IA Local, Offline-First
#
#  Copyright (c) 2026 Wil-1302
#  Licensed under the MIT License
#
#  Upstream origin: brenpoly/be-more-agent (MIT License)
#  Adaptado significativamente para Arch Linux como asistente terminal-first.
#
#  DISCLAIMER:
#  This software is provided "as is", without warranty of any kind.
# =========================================================================

import threading
import time
import json
import os
import subprocess
import random
import re
import sys
import select
import traceback
import atexit
import datetime
import warnings
import wave

warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")

# =========================================================================
# PROJECT ROOT
# =========================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def _p(*parts):
    return os.path.join(PROJECT_ROOT, *parts)

# Core dependencies
import sounddevice as sd
import numpy as np
import scipy.signal

# AI engines
import openwakeword
from openwakeword.model import Model
import ollama

# Web search
from ddgs import DDGS

# =========================================================================
# 1. CONFIGURACIÓN
# =========================================================================

CONFIG_FILE     = _p("config.json")
MEMORY_FILE     = _p("memory.json")
WAKE_WORD_MODEL = _p("wakeword.onnx")
WAKE_WORD_THRESHOLD = 0.5

WHISPER_CLI = _p("whisper.cpp", "build", "bin", "whisper-cli")
PIPER_BIN   = _p("piper", "piper")

INPUT_DEVICE_NAME = None

DEFAULT_CONFIG = {
    "text_model": "gemma3:1b",
    "language": "es",
    # Recommended for Spanish: ggml-small.bin (much better accuracy, ~244MB)
    # Minimum viable: ggml-base.bin (~74MB, less accurate)
    "whisper_model": "ggml-small.bin",
    "whisper_threads": 4,
    "voice_model": _p("piper", "es_ES-davefx-medium.onnx"),
    "chat_memory": True,
    "system_prompt_extras": "",
    # Audio tuning
    "silence_threshold": 0.015,   # RMS threshold (0.0–1.0) — raise if room noise keeps recording
    "silence_duration": 1.5,      # seconds of silence before stopping
    "min_speech_duration": 0.5,   # seconds of speech required before silence detection activates
    "max_record_time": 30.0,
    # noise_floor_multiplier: effective threshold = max(silence_threshold, noise_floor * multiplier)
    "noise_floor_multiplier": 2.0,
    # Debug: save last recording to last_recording.wav for manual inspection
    "debug_audio": True,
    # Wake word: set to true to enable "hey axis" trigger; false = Enter/PTT only (recommended for testing)
    "use_wake_word": False,
}

OLLAMA_OPTIONS = {
    'keep_alive': '-1',
    'num_thread': 4,
    'temperature': 0.7,
    'top_k': 40,
    'top_p': 0.9
}

def load_config():
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config.update(json.load(f))
        except Exception as e:
            print(f"[CONFIG] Error al cargar config.json: {e}. Usando valores por defecto.")
    return config

CURRENT_CONFIG = load_config()
TEXT_MODEL    = CURRENT_CONFIG["text_model"]
LANGUAGE      = CURRENT_CONFIG.get("language", "es")
WHISPER_MODEL = _p("whisper.cpp", "models", CURRENT_CONFIG.get("whisper_model", "ggml-small.bin"))
SEARCH_REGION = "es-es" if LANGUAGE == "es" else "us-en"

# Audio tuning — read once from config
SILENCE_THRESHOLD    = float(CURRENT_CONFIG.get("silence_threshold", 0.015))
SILENCE_DURATION     = float(CURRENT_CONFIG.get("silence_duration", 1.5))
MIN_SPEECH_DURATION  = float(CURRENT_CONFIG.get("min_speech_duration", 0.5))
MAX_RECORD_TIME      = float(CURRENT_CONFIG.get("max_record_time", 30.0))
NOISE_FLOOR_MULT     = float(CURRENT_CONFIG.get("noise_floor_multiplier", 2.0))
WHISPER_THREADS      = int(CURRENT_CONFIG.get("whisper_threads", 4))
DEBUG_AUDIO          = bool(CURRENT_CONFIG.get("debug_audio", True))
USE_WAKE_WORD        = bool(CURRENT_CONFIG.get("use_wake_word", False))

# Fixed target rate for Whisper (it internally works at 16 kHz)
WHISPER_SAMPLE_RATE = 16000

# --- SYSTEM PROMPT ---
# NOTE: capture_image is intentionally excluded from available tools in this
# terminal phase. Camera support is preserved in agent_legacy_gui.py for
# a future phase.
_FALLBACK_SYSTEM_PROMPT = (
    "Eres A.X.I.S. (Autonomous eXtended Intelligence System), un asistente de IA local. "
    "Responde SIEMPRE en español, con frases cortas y claras.\n\n"
    "Tienes acceso a las siguientes herramientas. Para usarlas, responde ÚNICAMENTE con el JSON indicado:\n\n"
    '1. Ver la hora: {"action": "get_time"}\n'
    '2. Buscar en internet: {"action": "search_web", "query": "término de búsqueda"}\n\n'
    "Para conversación normal, responde con texto en español. Nunca respondas en inglés."
)

if CURRENT_CONFIG.get("system_prompt"):
    SYSTEM_PROMPT = CURRENT_CONFIG["system_prompt"]
else:
    SYSTEM_PROMPT = _FALLBACK_SYSTEM_PROMPT + "\n\n" + CURRENT_CONFIG.get("system_prompt_extras", "")

# Sound directories
greeting_sounds_dir = _p("sounds", "greeting_sounds")
ack_sounds_dir      = _p("sounds", "ack_sounds")
thinking_sounds_dir = _p("sounds", "thinking_sounds")
error_sounds_dir    = _p("sounds", "error_sounds")

class BotStates:
    IDLE      = "idle"
    LISTENING = "listening"
    THINKING  = "thinking"
    SPEAKING  = "speaking"
    ERROR     = "error"
    WARMUP    = "warmup"

# =========================================================================
# 2. TERMINAL OUTPUT HELPERS
# =========================================================================

# ANSI color codes
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BLUE   = "\033[94m"
    GRAY   = "\033[90m"
    WHITE  = "\033[97m"

def banner():
    print(f"\n{C.CYAN}{C.BOLD}", flush=True)
    print("  ╔═══════════════════════════════════════╗", flush=True)
    print("  ║   A . X . I . S .                    ║", flush=True)
    print("  ║   Autonomous eXtended Intelligence   ║", flush=True)
    print("  ║   System  —  Arch Linux  —  v0.8     ║", flush=True)
    print("  ╚═══════════════════════════════════════╝", flush=True)
    print(f"{C.RESET}", flush=True)

STATE_LABELS = {
    BotStates.IDLE:      f"{C.GRAY}[IDLE]{C.RESET}",
    BotStates.LISTENING: f"{C.GREEN}[ESCUCHANDO]{C.RESET}",
    BotStates.THINKING:  f"{C.YELLOW}[PENSANDO]{C.RESET}",
    BotStates.SPEAKING:  f"{C.BLUE}[HABLANDO]{C.RESET}",
    BotStates.ERROR:     f"{C.RED}[ERROR]{C.RESET}",
    BotStates.WARMUP:    f"{C.CYAN}[INICIANDO]{C.RESET}",
}

# =========================================================================
# 3. AXIS TERMINAL CLASS
# =========================================================================

class AxisTerminal:
    def __init__(self):
        self.current_state = BotStates.WARMUP
        self.current_volume = 0

        self.permanent_memory = self.load_chat_history()
        self.session_memory   = []

        self.thinking_sound_active = threading.Event()
        self.ptt_event             = threading.Event()
        self.recording_active      = threading.Event()
        self.interrupted           = threading.Event()

        self.tts_queue      = []
        self.tts_queue_lock = threading.Lock()
        self.tts_thread     = None
        self.tts_active     = threading.Event()
        self.current_audio_process = None

        atexit.register(self.safe_exit)

        # Wake word — disabled by default for testing (set use_wake_word: true in config.json to enable)
        self.oww_model = None
        if not USE_WAKE_WORD:
            print(f"{C.YELLOW}[INIT] Wake word desactivado (use_wake_word=false) — modo PTT/Enter activo.{C.RESET}", flush=True)
        elif not os.path.exists(WAKE_WORD_MODEL):
            print(f"{C.YELLOW}[AVISO] No se encontró wakeword.onnx — usando modo PTT (Enter).{C.RESET}", flush=True)
        else:
            print(f"{C.GRAY}[INIT] Cargando modelo de palabra de activación...{C.RESET}", flush=True)
            try:
                self.oww_model = Model(wakeword_model_paths=[WAKE_WORD_MODEL])
                print(f"{C.GREEN}[INIT] Wake word cargado.{C.RESET}", flush=True)
            except TypeError:
                try:
                    self.oww_model = Model(wakeword_models=[WAKE_WORD_MODEL])
                    print(f"{C.GREEN}[INIT] Wake word cargado (API nueva).{C.RESET}", flush=True)
                except Exception as e:
                    print(f"{C.RED}[CRÍTICO] Fallo al cargar wake word: {e}{C.RESET}", flush=True)
            except Exception as e:
                print(f"{C.RED}[CRÍTICO] Fallo al cargar wake word: {e}{C.RESET}", flush=True)

    # -------------------------------------------------------------------------
    # Terminal I/O
    # -------------------------------------------------------------------------

    def set_state(self, state, msg=""):
        self.current_state = state
        label = STATE_LABELS.get(state, f"[{state.upper()}]")
        if msg:
            print(f"  {label} {msg}", flush=True)

    def print_user(self, text):
        print(f"\n  {C.WHITE}{C.BOLD}TÚ:{C.RESET}    {text}", flush=True)

    def print_axis(self, text="", end="\n"):
        if text == "":
            print(flush=True)
            return
        sys.stdout.write(f"\r  {C.CYAN}{C.BOLD}A.X.I.S.:{C.RESET} {text}" + end)
        sys.stdout.flush()

    def stream_chunk(self, chunk):
        sys.stdout.write(chunk)
        sys.stdout.flush()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def extract_json_from_text(self, text):
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return None
        except (json.JSONDecodeError, AttributeError):
            return None

    def safe_exit(self):
        print(f"\n{C.GRAY}--- A.X.I.S. CERRANDO ---{C.RESET}", flush=True)
        if self.current_audio_process:
            try:
                self.current_audio_process.terminate()
                self.current_audio_process.wait(timeout=1)
            except Exception:
                pass
        self.recording_active.clear()
        self.thinking_sound_active.clear()
        self.tts_active.clear()
        self.save_chat_history()
        try:
            ollama.generate(model=TEXT_MODEL, prompt="", keep_alive=0)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Action router (camera disabled for terminal phase)
    # -------------------------------------------------------------------------

    def execute_action_and_get_result(self, action_data):
        raw_action = action_data.get("action", "").lower().strip()
        value = action_data.get("value") or action_data.get("query")

        # Camera is intentionally excluded in this terminal phase.
        VALID_TOOLS = {"get_time", "search_web"}

        ALIASES = {
            "google": "search_web", "browser": "search_web", "news": "search_web",
            "search_news": "search_web", "check_time": "get_time",
            # Camera aliases → redirect to conversational fallback
            "capture_image": "CAMERA_DISABLED",
            "look": "CAMERA_DISABLED", "see": "CAMERA_DISABLED",
        }

        action = ALIASES.get(raw_action, raw_action)

        if action == "CAMERA_DISABLED":
            return "CHAT_FALLBACK::Lo siento, la cámara no está disponible en este momento."

        if action not in VALID_TOOLS:
            if value and isinstance(value, str) and len(value.split()) > 1:
                return f"CHAT_FALLBACK::{value}"
            return "INVALID_ACTION"

        if action == "get_time":
            now = datetime.datetime.now().strftime("%H:%M")
            return f"Son las {now}."

        elif action == "search_web":
            print(f"  {C.GRAY}[WEB] Buscando: {value}...{C.RESET}", flush=True)
            try:
                with DDGS() as ddgs:
                    results = []
                    try:
                        results = list(ddgs.news(value, region=SEARCH_REGION, max_results=1))
                    except Exception:
                        pass
                    if not results:
                        try:
                            results = list(ddgs.text(value, region=SEARCH_REGION, max_results=1))
                        except Exception:
                            pass
                    if results:
                        r = results[0]
                        title = r.get('title', 'Sin título')
                        body  = r.get('body', r.get('snippet', ''))
                        return f"SEARCH RESULTS for '{value}':\nTitle: {title}\nSnippet: {body[:300]}"
                    return "SEARCH_EMPTY"
            except Exception as e:
                print(f"  {C.GRAY}[WEB] Error de conexión: {e}{C.RESET}", flush=True)
                return "SEARCH_ERROR"

        return None

    # -------------------------------------------------------------------------
    # Core loop
    # -------------------------------------------------------------------------

    def run(self):
        self.warm_up_logic()
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()

        while True:
            try:
                trigger_source = self.detect_wake_word_or_ptt()
                if self.interrupted.is_set():
                    self.interrupted.clear()
                    self.set_state(BotStates.IDLE, "Reiniciando...")
                    continue

                self.set_state(BotStates.LISTENING, "¡Te escucho!")

                if trigger_source == "PTT":
                    audio_file = self.record_voice_ptt()
                else:
                    audio_file = self.record_voice_adaptive()

                if not audio_file:
                    self.set_state(BotStates.IDLE, "No escuché nada.")
                    continue

                user_text = self.transcribe_audio(audio_file)
                if not user_text:
                    self.set_state(BotStates.IDLE, "Transcripción vacía.")
                    continue

                self.print_user(user_text)
                self.interrupted.clear()
                self.chat_and_respond(user_text)

            except KeyboardInterrupt:
                print(f"\n{C.GRAY}Interrumpido por el usuario.{C.RESET}")
                self.safe_exit()
                sys.exit(0)
            except Exception as e:
                traceback.print_exc()
                self.set_state(BotStates.ERROR, f"Error fatal: {str(e)[:60]}")
                time.sleep(2)

    def warm_up_logic(self):
        self.set_state(BotStates.WARMUP, "Calentando modelos...")
        try:
            ollama.generate(model=TEXT_MODEL, prompt="", keep_alive=-1)
        except Exception as e:
            print(f"  {C.YELLOW}[AVISO] Error cargando {TEXT_MODEL}: {e}{C.RESET}", flush=True)
        self.play_sound(self.get_random_sound(greeting_sounds_dir))
        print(f"  {C.GREEN}[INIT] Modelos cargados. A.X.I.S. listo.{C.RESET}\n", flush=True)

    def detect_wake_word_or_ptt(self):
        self.set_state(BotStates.IDLE, "Esperando activación...")
        self.ptt_event.clear()

        if self.oww_model:
            self.oww_model.reset()

        if self.oww_model is None:
            # PTT mode: block on stdin until the user presses Enter.
            # ptt_event is kept for optional external triggers (hotkey, signal).
            # If ptt_event is already set (external trigger), use it immediately.
            if not self.ptt_event.is_set():
                sys.stdout.write(
                    f"\n  {C.GRAY}──────────────────────────────────────────────\n"
                    f"  [ Enter para hablar  |  Ctrl+C para salir ]\n"
                    f"  ──────────────────────────────────────────────{C.RESET}\n"
                )
                sys.stdout.flush()
                try:
                    sys.stdin.readline()
                except (EOFError, KeyboardInterrupt):
                    raise KeyboardInterrupt
            self.ptt_event.clear()
            return "PTT"

        CHUNK_SIZE     = 1280
        OWW_SAMPLE_RATE = 16000

        try:
            device_info = sd.query_devices(kind='input')
            native_rate = int(device_info['default_samplerate'])
        except Exception:
            native_rate = 48000

        use_resampling   = (native_rate != OWW_SAMPLE_RATE)
        input_rate       = native_rate if use_resampling else OWW_SAMPLE_RATE
        input_chunk_size = int(CHUNK_SIZE * (input_rate / OWW_SAMPLE_RATE)) if use_resampling else CHUNK_SIZE

        try:
            with sd.InputStream(samplerate=input_rate, channels=1, dtype='int16',
                                 blocksize=input_chunk_size, device=INPUT_DEVICE_NAME) as stream:
                while True:
                    if self.ptt_event.is_set():
                        self.ptt_event.clear()
                        return "PTT"

                    rlist, _, _ = select.select([sys.stdin], [], [], 0.001)
                    if rlist:
                        sys.stdin.readline()
                        return "CLI"

                    data, _ = stream.read(input_chunk_size)
                    audio_data = np.frombuffer(data, dtype=np.int16)

                    if use_resampling:
                        audio_data = scipy.signal.resample(audio_data, CHUNK_SIZE).astype(np.int16)

                    self.oww_model.predict(audio_data)
                    for mdl in self.oww_model.prediction_buffer.keys():
                        if list(self.oww_model.prediction_buffer[mdl])[-1] > WAKE_WORD_THRESHOLD:
                            self.oww_model.reset()
                            return "WAKE"
        except Exception as e:
            print(f"  {C.YELLOW}[AVISO] Error en detección de wake word: {e}{C.RESET}", flush=True)
            # Wake word stream failed — fall back to PTT mode for this turn.
            sys.stdout.write(
                f"\n  {C.GRAY}──────────────────────────────────────────────\n"
                f"  [ Enter para hablar  |  Ctrl+C para salir ]\n"
                f"  ──────────────────────────────────────────────{C.RESET}\n"
            )
            sys.stdout.flush()
            try:
                sys.stdin.readline()
            except (EOFError, KeyboardInterrupt):
                raise KeyboardInterrupt
            return "PTT"

    def record_voice_adaptive(self, filename="input.wav"):
        self.set_state(BotStates.LISTENING, "Grabando (detección de silencio)...")
        time.sleep(0.3)  # brief gap after state change sound

        try:
            samplerate = int(sd.query_devices(kind='input')['default_samplerate'])
        except Exception:
            samplerate = 44100

        chunk_duration    = 0.05  # 50 ms chunks
        chunk_size        = int(samplerate * chunk_duration)
        num_silent_chunks = int(SILENCE_DURATION / chunk_duration)
        max_chunks        = int(MAX_RECORD_TIME / chunk_duration)
        min_speech_chunks = int(MIN_SPEECH_DURATION / chunk_duration)

        # --- Sample ambient noise floor to compute an adaptive threshold ---
        effective_threshold = SILENCE_THRESHOLD
        try:
            noise_samples = []
            with sd.InputStream(samplerate=samplerate, channels=1, dtype='float32',
                                 blocksize=chunk_size, device=INPUT_DEVICE_NAME) as ns:
                for _ in range(10):  # ~500 ms
                    data, _ = ns.read(chunk_size)
                    noise_samples.append(np.linalg.norm(data) / np.sqrt(len(data)))
            noise_floor = float(np.mean(noise_samples))
            effective_threshold = max(SILENCE_THRESHOLD, noise_floor * NOISE_FLOOR_MULT)
            print(f"  {C.GRAY}[MIC] Tasa: {samplerate} Hz | "
                  f"Ruido ambiente: {noise_floor:.4f} | "
                  f"Umbral efectivo: {effective_threshold:.4f} | "
                  f"Silencio tras: {SILENCE_DURATION:.1f}s{C.RESET}", flush=True)
        except Exception:
            print(f"  {C.GRAY}[MIC] Tasa: {samplerate} Hz | "
                  f"Umbral: {effective_threshold:.4f} (base) | "
                  f"Silencio tras: {SILENCE_DURATION:.1f}s{C.RESET}", flush=True)

        print(f"  {C.GREEN}[MIC] Grabando... (habla ahora){C.RESET}", flush=True)

        buffer           = []
        silent_chunks    = 0
        recorded_chunks  = 0
        speech_chunks    = 0
        silence_started  = False
        record_start     = time.time()

        def callback(indata, frames, time_info, status):
            nonlocal silent_chunks, recorded_chunks, speech_chunks, silence_started
            if status:
                print(f"  {C.GRAY}[MIC] {status}{C.RESET}", flush=True)
            volume_rms = np.linalg.norm(indata) / np.sqrt(len(indata))
            buffer.append(indata.copy())
            recorded_chunks += 1

            if volume_rms >= effective_threshold:
                speech_chunks += 1
                silent_chunks  = 0
            else:
                # Only start counting silence once minimum speech is detected
                if speech_chunks >= min_speech_chunks:
                    silent_chunks += 1
                    if silent_chunks >= num_silent_chunks:
                        silence_started = True

        try:
            with sd.InputStream(samplerate=samplerate, channels=1, dtype='float32',
                                 callback=callback, device=INPUT_DEVICE_NAME,
                                 blocksize=chunk_size):
                while not silence_started and recorded_chunks < max_chunks:
                    sd.sleep(int(chunk_duration * 1000))
        except Exception as e:
            print(f"  {C.RED}[AUDIO] Error de grabación: {e}{C.RESET}", flush=True)
            return None

        duration = time.time() - record_start
        if silence_started:
            stop_reason = f"{C.GREEN}silencio detectado{C.RESET}"
        else:
            stop_reason = f"{C.YELLOW}tiempo máximo ({MAX_RECORD_TIME:.0f}s){C.RESET}"
        print(f"  {C.GRAY}[MIC] Grabación terminada ({stop_reason}{C.GRAY}) — "
              f"{duration:.2f}s | {speech_chunks} chunks habla | "
              f"{silent_chunks} chunks silencio final{C.RESET}", flush=True)

        if speech_chunks < min_speech_chunks:
            print(f"  {C.YELLOW}[MIC] Habla insuficiente detectada — ignorando.{C.RESET}", flush=True)
            return None

        return self.save_audio_buffer(buffer, filename, samplerate)

    def record_voice_ptt(self, filename="input.wav"):
        self.set_state(BotStates.LISTENING, "Grabando (PTT — Enter para detener)...")
        time.sleep(0.3)

        try:
            samplerate = int(sd.query_devices(kind='input')['default_samplerate'])
        except Exception:
            samplerate = 44100

        print(f"  {C.GRAY}[MIC] Tasa: {samplerate} Hz{C.RESET}", flush=True)
        print(f"  {C.GREEN}[MIC] Grabando... (presiona Enter para detener){C.RESET}", flush=True)

        buffer     = []
        stop_event = threading.Event()
        record_start = time.time()

        def callback(indata, frames, time_info, status):
            if status:
                print(f"  {C.GRAY}[MIC] {status}{C.RESET}", flush=True)
            buffer.append(indata.copy())

        def wait_for_enter():
            input()
            stop_event.set()

        enter_thread = threading.Thread(target=wait_for_enter, daemon=True)
        enter_thread.start()

        try:
            with sd.InputStream(samplerate=samplerate, channels=1, dtype='float32',
                                 callback=callback, device=INPUT_DEVICE_NAME):
                while not stop_event.is_set():
                    sd.sleep(50)
        except Exception as e:
            print(f"  {C.RED}[AUDIO] Error de grabación PTT: {e}{C.RESET}", flush=True)
            return None

        duration = time.time() - record_start
        print(f"  {C.GRAY}[MIC] PTT terminado — {duration:.2f}s grabados{C.RESET}", flush=True)
        return self.save_audio_buffer(buffer, filename, samplerate)

    def save_audio_buffer(self, buffer, filename, samplerate=44100):
        if not buffer:
            return None

        audio_data = np.concatenate(buffer, axis=0).flatten()
        audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)

        # Resample to 16 kHz — Whisper.cpp works natively at 16 kHz.
        # Providing pre-resampled audio avoids internal resampling artifacts.
        if samplerate != WHISPER_SAMPLE_RATE:
            num_target = int(len(audio_data) * WHISPER_SAMPLE_RATE / samplerate)
            audio_data = scipy.signal.resample(audio_data, num_target)

        audio_int16 = (audio_data * 32767).astype(np.int16)
        duration_s  = len(audio_int16) / WHISPER_SAMPLE_RATE
        filepath    = _p(filename)

        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(WHISPER_SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())

        print(f"  {C.GRAY}[AUDIO] Guardado: {filepath} "
              f"| {WHISPER_SAMPLE_RATE} Hz | {duration_s:.2f}s{C.RESET}", flush=True)

        # Save a debug copy for manual inspection (aplay last_recording.wav)
        if DEBUG_AUDIO:
            debug_path = _p("last_recording.wav")
            try:
                import shutil
                shutil.copy2(filepath, debug_path)
            except Exception:
                pass

        self.play_sound(self.get_random_sound(ack_sounds_dir))
        return filepath

    def clean_transcription(self, text):
        """Strip whisper artifacts and noise markers from raw transcript text."""
        # Remove special tokens: [_EOT_], [_BEG_], [_TT_NNN], [BLANK_AUDIO], etc.
        text = re.sub(r'\[_[A-Z0-9_]+\]', '', text)
        text = re.sub(r'\[BLANK_AUDIO\]', '', text, flags=re.IGNORECASE)
        # Remove music / sound markers (♪ ♫) and parenthetical noise tags like (music)
        text = re.sub(r'[♪♫🎵🎶]', '', text)
        text = re.sub(r'\([^)]{1,40}\)', '', text)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def transcribe_audio(self, filename):
        self.set_state(BotStates.THINKING, "Transcribiendo audio...")

        if not os.path.exists(WHISPER_CLI):
            print(f"  {C.RED}[STT] whisper-cli no encontrado: {WHISPER_CLI}{C.RESET}", flush=True)
            print(f"  {C.RED}[STT] Solución: ejecuta setup.sh para compilar whisper.cpp{C.RESET}", flush=True)
            return ""
        if not os.path.exists(WHISPER_MODEL):
            print(f"  {C.RED}[STT] Modelo no encontrado: {WHISPER_MODEL}{C.RESET}", flush=True)
            model_name = os.path.basename(WHISPER_MODEL)
            print(f"  {C.RED}[STT] Solución: descarga {model_name} en whisper.cpp/models/{C.RESET}", flush=True)
            if "small" in model_name:
                print(f"  {C.YELLOW}[STT] Tip: wget -O whisper.cpp/models/{model_name} "
                      f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{model_name}{C.RESET}", flush=True)
            return ""

        cmd = [
            WHISPER_CLI,
            "-m", WHISPER_MODEL,
            "-f", filename,
            "-l", LANGUAGE,          # force Spanish
            "-t", str(WHISPER_THREADS),
            "--no-timestamps",       # cleaner output, easier to parse
            # NOTE: do NOT pass --print-special; omitting it keeps special tokens suppressed by default.
        ]

        print(f"  {C.GRAY}[STT] Comando: {' '.join(cmd)}{C.RESET}", flush=True)
        t_start = time.time()

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            elapsed = time.time() - t_start

            if result.returncode != 0:
                print(f"  {C.RED}[STT] whisper-cli falló (código {result.returncode}){C.RESET}", flush=True)
                if result.stderr.strip():
                    print(f"  {C.RED}[STT] stderr: {result.stderr.strip()}{C.RESET}", flush=True)
                return ""

            raw_output = result.stdout.strip()

            if DEBUG_AUDIO and raw_output:
                print(f"  {C.GRAY}[STT] Salida bruta ({elapsed:.1f}s):{C.RESET}", flush=True)
                for line in raw_output.splitlines():
                    print(f"    {C.GRAY}{line}{C.RESET}", flush=True)

            # Collect all non-empty transcript lines.
            # With --no-timestamps, output looks like plain text lines.
            # Without it, lines look like: [00:00:00.000 --> 00:00:03.000]  text
            # We handle both formats for robustness.
            transcript_parts = []
            for line in raw_output.splitlines():
                line = line.strip()
                if not line:
                    continue
                # Skip whisper.cpp header/info lines (they start with 'whisper_' or 'system_')
                if re.match(r'^(whisper_|system_|main:|\[INST\])', line):
                    continue
                # If timestamps were included despite --no-timestamps (older builds),
                # strip the timestamp prefix: [00:00:00.000 --> 00:00:03.000]
                line = re.sub(r'^\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}\]\s*', '', line)
                # Skip lines that are just special tokens
                if re.match(r'^\[.*\]$', line):
                    continue
                if line:
                    transcript_parts.append(line)

            transcription = " ".join(transcript_parts).strip()
            # Strip whisper artifacts and normalize whitespace
            transcription = self.clean_transcription(transcription)

            if transcription:
                print(f"  {C.GREEN}[STT] Transcripción: \"{transcription}\"{C.RESET}", flush=True)
            else:
                print(f"  {C.YELLOW}[STT] Sin transcripción — audio vacío o inaudible.{C.RESET}", flush=True)
                if not DEBUG_AUDIO and raw_output:
                    # Show raw output on failure even if debug is off
                    print(f"  {C.GRAY}[STT] Salida bruta: {raw_output[:300]}{C.RESET}", flush=True)

            return transcription

        except subprocess.TimeoutExpired:
            print(f"  {C.RED}[STT] whisper-cli excedió el tiempo de espera (60s){C.RESET}", flush=True)
            print(f"  {C.YELLOW}[STT] Tip: usa un modelo más pequeño o reduce --threads{C.RESET}", flush=True)
            return ""
        except Exception as e:
            print(f"  {C.RED}[STT] Error inesperado: {e}{C.RESET}", flush=True)
            traceback.print_exc()
            return ""

    # -------------------------------------------------------------------------
    # Chat & Respond
    # -------------------------------------------------------------------------

    def chat_and_respond(self, text):
        _lower = text.lower()
        if any(t in _lower for t in ("olvida todo", "borrar memoria", "forget everything", "reset memory")):
            self.session_memory   = []
            self.permanent_memory = [{"role": "system", "content": SYSTEM_PROMPT}]
            self.save_chat_history()
            msg = "De acuerdo. Memoria borrada."
            self.print_axis(msg)
            with self.tts_queue_lock:
                self.tts_queue.append(msg)
            self.set_state(BotStates.IDLE, "Memoria borrada")
            return

        self.set_state(BotStates.THINKING, "Pensando...")

        # Inject a per-turn Spanish reminder for small models that may drift to English.
        # The reminder is sent to the LLM but NOT stored in memory or shown to the user.
        if LANGUAGE == "es":
            llm_content = text + "\n\n(IMPORTANTE: responde ÚNICAMENTE en español, nunca en inglés.)"
        else:
            llm_content = text
        user_msg = {"role": "user", "content": llm_content}
        messages = self.permanent_memory + self.session_memory + [user_msg]

        self.thinking_sound_active.set()
        threading.Thread(target=self._run_thinking_sound_loop, daemon=True).start()

        full_response_buffer = ""
        sentence_buffer      = ""
        axis_label_printed   = False

        try:
            stream = ollama.chat(model=TEXT_MODEL, messages=messages, stream=True, options=OLLAMA_OPTIONS)
            is_action_mode = False

            for chunk in stream:
                if self.interrupted.is_set():
                    break
                content = chunk['message']['content']
                full_response_buffer += content

                # Detect action mode by checking whether the response STARTS with JSON.
                # Only switch before we have printed any output — prevents false triggers
                # on mid-sentence braces in normal prose.
                if not axis_label_printed and not is_action_mode:
                    stripped_so_far = full_response_buffer.lstrip()
                    if stripped_so_far.startswith('{'):
                        is_action_mode = True
                        self.thinking_sound_active.clear()
                        continue

                if is_action_mode:
                    continue

                self.thinking_sound_active.clear()
                if self.current_state != BotStates.SPEAKING:
                    self.set_state(BotStates.SPEAKING, "")
                    sys.stdout.write(f"\n  {C.CYAN}{C.BOLD}A.X.I.S.:{C.RESET} ")
                    sys.stdout.flush()
                    axis_label_printed = True

                self.stream_chunk(content)

                sentence_buffer += content
                if any(punct in content for punct in ".!?\n"):
                    clean = sentence_buffer.strip()
                    if clean and re.search(r'\w', clean):
                        with self.tts_queue_lock:
                            self.tts_queue.append(clean)
                    sentence_buffer = ""

            # Flush any remaining sentence fragment that had no trailing punctuation.
            if not is_action_mode and sentence_buffer.strip() and re.search(r'\w', sentence_buffer.strip()):
                with self.tts_queue_lock:
                    self.tts_queue.append(sentence_buffer.strip())
            sentence_buffer = ""

            if axis_label_printed:
                print(flush=True)  # newline after streamed response

            if is_action_mode:
                action_data = self.extract_json_from_text(full_response_buffer)
                if action_data:
                    tool_result = self.execute_action_and_get_result(action_data)

                    if tool_result and tool_result.startswith("CHAT_FALLBACK::"):
                        response_text = tool_result.split("::", 1)[1]
                        self.thinking_sound_active.clear()
                        self.set_state(BotStates.SPEAKING, "")
                        self.print_axis(response_text)
                        with self.tts_queue_lock:
                            self.tts_queue.append(response_text)
                        self.session_memory.append({"role": "assistant", "content": response_text})
                        self.wait_for_tts()
                        self.set_state(BotStates.IDLE, "Listo")
                        return

                    elif tool_result == "INVALID_ACTION":
                        fallback = "No sé cómo hacer eso."
                        self.thinking_sound_active.clear()
                        self.print_axis(fallback)
                        with self.tts_queue_lock:
                            self.tts_queue.append(fallback)

                    elif tool_result == "SEARCH_EMPTY":
                        fallback = "Busqué pero no encontré resultados."
                        self.thinking_sound_active.clear()
                        self.print_axis(fallback)
                        with self.tts_queue_lock:
                            self.tts_queue.append(fallback)

                    elif tool_result == "SEARCH_ERROR":
                        fallback = "Ahora mismo no puedo acceder a internet."
                        self.thinking_sound_active.clear()
                        self.print_axis(fallback)
                        with self.tts_queue_lock:
                            self.tts_queue.append(fallback)

                    elif tool_result:
                        summary_prompt = [
                            {"role": "system", "content": "Eres un asistente que SIEMPRE responde en español. Resume el siguiente resultado en UNA frase corta en español. NUNCA respondas en inglés."},
                            {"role": "user",   "content": f"RESULTADO: {tool_result}\nPregunta original: {text}\n\n(Responde en español.)"}
                        ]
                        self.set_state(BotStates.THINKING, "Leyendo resultados...")
                        self.thinking_sound_active.set()

                        final_resp = ollama.chat(model=TEXT_MODEL, messages=summary_prompt,
                                                 stream=False, options=OLLAMA_OPTIONS)
                        final_text = final_resp['message']['content']

                        self.thinking_sound_active.clear()
                        self.set_state(BotStates.SPEAKING, "")
                        self.print_axis(final_text)
                        with self.tts_queue_lock:
                            self.tts_queue.append(final_text)
                        self.session_memory.append({"role": "assistant", "content": final_text})
            else:
                self.session_memory.append({"role": "assistant", "content": full_response_buffer})

            self.wait_for_tts()
            self.set_state(BotStates.IDLE, "Listo")

        except Exception as e:
            print(f"\n  {C.RED}[LLM] Error: {e}{C.RESET}", flush=True)
            self.set_state(BotStates.ERROR, "Error de procesamiento")

    def wait_for_tts(self):
        while self.tts_queue or self.tts_active.is_set():
            if self.interrupted.is_set():
                break
            time.sleep(0.1)

    # -------------------------------------------------------------------------
    # TTS
    # -------------------------------------------------------------------------

    def _tts_worker(self):
        while True:
            text = None
            with self.tts_queue_lock:
                if self.tts_queue:
                    text = self.tts_queue.pop(0)
                    self.tts_active.set()
                else:
                    # Queue drained — mark inactive so wait_for_tts() can exit.
                    self.tts_active.clear()
            if text:
                self.speak(text)
            else:
                time.sleep(0.05)

    def speak(self, text):
        clean = re.sub(r"[^\w\s,.!?¡¿:;'\-]", "", text, flags=re.UNICODE)
        if not clean.strip():
            return

        if not os.path.exists(PIPER_BIN):
            return

        voice_model = CURRENT_CONFIG.get("voice_model", _p("piper", "es_ES-davefx-medium.onnx"))
        if not os.path.isabs(voice_model):
            voice_model = _p(voice_model)

        try:
            self.current_audio_process = subprocess.Popen(
                [PIPER_BIN, "--model", voice_model, "--output-raw"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.current_audio_process.stdin.write(clean.encode() + b'\n')
            self.current_audio_process.stdin.close()

            try:
                native_rate = int(sd.query_devices(kind='output')['default_samplerate'])
            except Exception:
                native_rate = 48000

            PIPER_RATE = 22050
            use_native_rate = False
            try:
                sd.check_output_settings(device=None, samplerate=PIPER_RATE)
            except Exception:
                use_native_rate = True

            with sd.RawOutputStream(samplerate=native_rate if use_native_rate else PIPER_RATE,
                                     channels=1, dtype='int16',
                                     device=None, latency='low', blocksize=2048) as stream:
                while True:
                    if self.interrupted.is_set():
                        break
                    data = self.current_audio_process.stdout.read(4096)
                    if not data:
                        break
                    audio_chunk = np.frombuffer(data, dtype=np.int16)
                    if len(audio_chunk) > 0:
                        if use_native_rate:
                            num_samples = int(len(audio_chunk) * (native_rate / PIPER_RATE))
                            audio_chunk = scipy.signal.resample(audio_chunk, num_samples).astype(np.int16)
                        stream.write(audio_chunk.tobytes())
            time.sleep(0.5)

        except Exception as e:
            print(f"  {C.GRAY}[TTS] Error: {e}{C.RESET}", flush=True)
        finally:
            self.current_volume = 0
            if self.current_audio_process:
                if self.current_audio_process.stderr:
                    err = self.current_audio_process.stderr.read()
                    if err:
                        errtxt = err.decode(errors='replace').strip()
                        if errtxt:
                            print(f"  {C.GRAY}[PIPER] {errtxt}{C.RESET}", flush=True)
                    self.current_audio_process.stderr.close()
                if self.current_audio_process.stdout:
                    self.current_audio_process.stdout.close()
                if self.current_audio_process.poll() is None:
                    self.current_audio_process.terminate()
                self.current_audio_process = None

    # -------------------------------------------------------------------------
    # Sound effects
    # -------------------------------------------------------------------------

    def _run_thinking_sound_loop(self):
        time.sleep(0.5)
        while self.thinking_sound_active.is_set():
            sound = self.get_random_sound(thinking_sounds_dir)
            if sound:
                self.play_sound(sound)
            for _ in range(50):
                if not self.thinking_sound_active.is_set():
                    return
                time.sleep(0.1)

    def get_random_sound(self, directory):
        if os.path.exists(directory):
            files = [f for f in os.listdir(directory) if f.endswith(".wav")]
            return os.path.join(directory, random.choice(files)) if files else None
        return None

    def play_sound(self, file_path):
        if not file_path or not os.path.exists(file_path):
            return
        try:
            with wave.open(file_path, 'rb') as wf:
                file_sr = wf.getframerate()
                data    = wf.readframes(wf.getnframes())
                audio   = np.frombuffer(data, dtype=np.int16)
            try:
                native_rate = int(sd.query_devices(kind='output')['default_samplerate'])
            except Exception:
                native_rate = 48000

            playback_rate = file_sr
            try:
                sd.check_output_settings(device=None, samplerate=file_sr)
            except Exception:
                playback_rate = native_rate
                num_samples   = int(len(audio) * (native_rate / file_sr))
                audio         = scipy.signal.resample(audio, num_samples).astype(np.int16)

            sd.play(audio, playback_rate)
            sd.wait()
        except Exception as e:
            pass  # Sound errors are non-fatal

    # -------------------------------------------------------------------------
    # Memory
    # -------------------------------------------------------------------------

    def load_chat_history(self):
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"  {C.GRAY}[MEMORIA] No se pudo cargar historial: {e}{C.RESET}", flush=True)
        return [{"role": "system", "content": SYSTEM_PROMPT}]

    def save_chat_history(self):
        full = self.permanent_memory + self.session_memory
        conv = full[1:]
        if len(conv) > 10:
            conv = conv[-10:]
        try:
            with open(MEMORY_FILE, "w") as f:
                json.dump([full[0]] + conv, f, indent=4)
        except Exception as e:
            print(f"  {C.GRAY}[MEMORIA] Error al guardar: {e}{C.RESET}", flush=True)


# =========================================================================
# 4. VERIFICACIÓN DE BINARIOS
# =========================================================================

def check_required_binaries():
    required = [
        (WHISPER_CLI,   "whisper-cli",     "Ejecuta setup.sh para compilar whisper.cpp"),
        (WHISPER_MODEL, "whisper model",   "Ejecuta setup.sh para descargar el modelo"),
        (PIPER_BIN,     "piper TTS",       "Ejecuta setup.sh para descargar piper"),
    ]
    missing = []
    for path, label, hint in required:
        if os.path.exists(path):
            print(f"  {C.GREEN}[OK]{C.RESET}    {label}: {C.GRAY}{path}{C.RESET}", flush=True)
        else:
            print(f"  {C.YELLOW}[FALTA]{C.RESET} {label}: {C.GRAY}{path}{C.RESET}", flush=True)
            print(f"         {C.GRAY}→ {hint}{C.RESET}", flush=True)
            missing.append(label)

    voice_model = CURRENT_CONFIG.get("voice_model", _p("piper", "es_ES-davefx-medium.onnx"))
    if not os.path.isabs(voice_model):
        voice_model = _p(voice_model)
    if os.path.exists(voice_model):
        print(f"  {C.GREEN}[OK]{C.RESET}    voice model: {C.GRAY}{voice_model}{C.RESET}", flush=True)
    else:
        print(f"  {C.YELLOW}[FALTA]{C.RESET} voice model: {C.GRAY}{voice_model}{C.RESET}", flush=True)
        print(f"         {C.GRAY}→ Ejecuta setup.sh para descargar el modelo de voz Piper{C.RESET}", flush=True)
        missing.append("voice model")

    if missing:
        print(f"\n  {C.YELLOW}[AVISO] {len(missing)} componente(s) faltante(s): {', '.join(missing)}{C.RESET}", flush=True)
        print(f"  {C.YELLOW}[AVISO] El flujo de voz no funcionará hasta que estén instalados.{C.RESET}\n", flush=True)
    return missing


# =========================================================================
# 5. ENTRYPOINT
# =========================================================================

if __name__ == "__main__":
    banner()
    print(f"  {C.GRAY}Raíz del proyecto : {PROJECT_ROOT}{C.RESET}", flush=True)
    print(f"  {C.GRAY}Idioma            : {LANGUAGE}  |  Región búsqueda: {SEARCH_REGION}{C.RESET}", flush=True)
    print(f"  {C.GRAY}Modelo LLM        : {TEXT_MODEL}{C.RESET}", flush=True)
    print(f"  {C.GRAY}Modelo Whisper    : {WHISPER_MODEL}{C.RESET}\n", flush=True)

    check_required_binaries()

    axis = AxisTerminal()
    axis.run()
