from __future__ import annotations

import importlib
import math
import os
import tempfile
import threading
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _import_optional(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise VoiceUnavailable(
            "语音输入需要可选依赖 sounddevice/faster-whisper。"
            "请在项目目录执行：.venv\\Scripts\\python -m pip install -e \".[voice]\""
        ) from exc

try:
    from platformdirs import user_data_dir
except ImportError:  # pragma: no cover - platformdirs is a hard dep in practice

    def user_data_dir(appname: str, appauthor: str | None = None) -> str:  # type: ignore[misc]
        return str(Path.home() / ".local" / "share" / appname)


SAMPLE_WIDTH = 2  # int16
DEFAULT_SAMPLE_RATE = 16_000
MIN_TRANSCRIBE_SECONDS = 0.35
SILENCE_RMS_THRESHOLD = 0.012  # ~ -38 dBFS on int16 normalized scale


class VoiceUnavailable(RuntimeError):
    """Raised when microphone or speech model cannot be used."""


def speech_cache_dir() -> Path:
    return Path(user_data_dir("CommanderTacticalArena", "CommanderTacticalArena")) / "speech"


def _int16_rms(samples: bytes) -> float:
    """Normalized RMS in 0..1 for little-endian int16 mono PCM."""
    if not samples:
        return 0.0
    count = len(samples) // SAMPLE_WIDTH
    if count == 0:
        return 0.0
    total = 0
    for i in range(0, count * SAMPLE_WIDTH, SAMPLE_WIDTH):
        value = int.from_bytes(samples[i : i + SAMPLE_WIDTH], "little", signed=True)
        total += value * value
    return math.sqrt(total / count) / 32768.0


def is_audible(samples: bytes, sample_rate: int = DEFAULT_SAMPLE_RATE) -> bool:
    if len(samples) < int(MIN_TRANSCRIBE_SECONDS * sample_rate) * SAMPLE_WIDTH:
        return False
    return _int16_rms(samples) >= SILENCE_RMS_THRESHOLD


class MicrophoneRecorder:
    """Hold-to-talk recorder. stop() is safe to call even if start() failed."""

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE, max_seconds: int = 15) -> None:
        self.sample_rate = sample_rate
        self.max_seconds = max_seconds
        self.max_samples = sample_rate * max_seconds
        self._chunks: list[bytes] = []
        self._bytes_captured = 0
        self._stream: Any = None
        self._lock = threading.Lock()
        self._level = 0.0

    @property
    def active(self) -> bool:
        return self._stream is not None

    @property
    def seconds(self) -> float:
        with self._lock:
            samples = self._bytes_captured // SAMPLE_WIDTH
        return samples / self.sample_rate if self.sample_rate else 0.0

    @property
    def level(self) -> float:
        """Latest chunk RMS in 0..1, for HUD meter."""
        return self._level

    def list_input_devices(self) -> list[str]:
        sounddevice = _import_optional("sounddevice")
        names: list[str] = []
        for device in sounddevice.query_devices():
            if int(device.get("max_input_channels", 0)) > 0:
                names.append(str(device.get("name", "输入设备")))
        return names

    def start(self, device: int | str | None = None) -> None:
        if self.active:
            return
        sounddevice = _import_optional("sounddevice")

        self._chunks = []
        self._bytes_captured = 0
        self._level = 0.0

        def callback(indata: Any, frames: int, time: Any, status: Any) -> None:
            del frames, time, status
            payload = bytes(indata)
            with self._lock:
                if self._bytes_captured >= self.max_samples * SAMPLE_WIDTH:
                    return
                room = self.max_samples * SAMPLE_WIDTH - self._bytes_captured
                chunk = payload[:room]
                if not chunk:
                    return
                self._chunks.append(chunk)
                self._bytes_captured += len(chunk)
                self._level = _int16_rms(chunk)

        try:
            self._stream = sounddevice.RawInputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                callback=callback,
                device=device,
            )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            self._chunks = []
            self._bytes_captured = 0
            raise VoiceUnavailable(f"无法打开麦克风：{exc}") from exc

    def stop(self) -> bytes:
        stream = self._stream
        self._stream = None
        self._level = 0.0
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        with self._lock:
            data = b"".join(self._chunks)[: self.max_samples * SAMPLE_WIDTH]
            self._chunks = []
            self._bytes_captured = 0
        return data

    def abort(self) -> None:
        """Discard capture without returning samples (window lost focus, cancel)."""
        self.stop()


class WhisperRecognizer:
    def __init__(
        self,
        model: str | Path = "small",
        allow_download: bool = False,
        language: str = "zh",
        cache_dir: Path | None = None,
    ) -> None:
        self.model_name = str(model)
        self.allow_download = allow_download
        self.language = language
        self.cache_dir = Path(cache_dir) if cache_dir is not None else speech_cache_dir()
        self._model: Any = None
        self._loading = False
        self._load_error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def loading(self) -> bool:
        return self._loading

    def resolve_model_path(self) -> Path | None:
        candidate = Path(self.model_name)
        if candidate.exists():
            return candidate
        # named model under cache (faster-whisper layout: cache/<name>)
        nested = self.cache_dir / self.model_name
        if nested.exists():
            return nested
        return None

    def status(self) -> dict[str, Any]:
        local = self.resolve_model_path()
        return {
            "model": self.model_name,
            "loaded": self.loaded,
            "loading": self._loading,
            "local_path": str(local) if local else None,
            "allow_download": self.allow_download,
            "error": self._load_error,
        }

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        if self._loading:
            raise VoiceUnavailable("语音模型正在加载，请稍候。")
        try:
            whisper_module = importlib.import_module("faster_whisper")
            WhisperModel = whisper_module.WhisperModel
        except ImportError as exc:
            self._load_error = "missing-dependency"
            raise VoiceUnavailable(
                "未安装 faster-whisper。可执行：pip install -e '.[voice]'"
            ) from exc

        local_path = self.resolve_model_path()
        if local_path is None and not self.allow_download:
            self._load_error = "needs-download"
            raise VoiceUnavailable("本地语音模型尚未准备；确认下载后再试。")

        self._loading = True
        self._load_error = None
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            load_target = str(local_path) if local_path is not None else self.model_name
            # When we only have a named model, prefer the shared cache directory.
            download_root = str(self.cache_dir) if local_path is None else None
            kwargs: dict[str, Any] = {
                "device": "cpu",
                "compute_type": "int8",
                "local_files_only": local_path is not None and not self.allow_download,
            }
            if download_root is not None:
                kwargs["download_root"] = download_root
            self._model = WhisperModel(load_target, **kwargs)
            return self._model
        except Exception as exc:
            self._load_error = "load-failed"
            raise VoiceUnavailable(f"语音模型加载失败：{exc}") from exc
        finally:
            self._loading = False

    def ensure_ready(self) -> None:
        """Load model without transcribing; used by settings / first confirm."""
        self._load()

    def transcribe(
        self,
        samples: bytes,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        *,
        require_audible: bool = True,
    ) -> str:
        if not samples:
            return ""
        if require_audible and not is_audible(samples, sample_rate):
            return ""
        model = self._load()
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                temporary = handle.name
            with wave.open(temporary, "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(SAMPLE_WIDTH)
                output.setframerate(sample_rate)
                output.writeframes(samples)
            segments, _info = model.transcribe(
                temporary,
                language=self.language,
                beam_size=3,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            text = "".join(segment.text for segment in segments).strip()
            return text
        finally:
            if temporary:
                Path(temporary).unlink(missing_ok=True)


def encode_wav(samples: bytes, sample_rate: int = DEFAULT_SAMPLE_RATE) -> bytes:
    """Encode raw int16 mono PCM into a WAV container in memory."""
    import io

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(SAMPLE_WIDTH)
        output.setframerate(sample_rate)
        output.writeframes(samples)
    return buffer.getvalue()


class OnlineSpeechRecognizer:
    """OpenAI-compatible speech-to-text (POST {base}/audio/transcriptions).

    Works with OpenAI, Groq, Azure OpenAI, SiliconFlow and similar gateways
    that expose the multipart Whisper API. Configure via env:
      SPEECH_API_KEY   required
      SPEECH_API_URL   default https://api.openai.com/v1
      SPEECH_API_MODEL default whisper-1
      SPEECH_API_LANG  default zh
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        language: str | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("SPEECH_API_KEY", "")
        raw_base = base_url if base_url is not None else os.getenv(
            "SPEECH_API_URL", "https://api.openai.com/v1"
        )
        self.base_url = (raw_base or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.getenv("SPEECH_API_MODEL", "whisper-1")
        self.language = language or os.getenv("SPEECH_API_LANG", "zh")
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "base_url": self.base_url,
            "model": self.model,
            "language": self.language,
        }

    def transcribe(
        self,
        samples: bytes,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        *,
        require_audible: bool = True,
    ) -> str:
        if not samples:
            return ""
        if require_audible and not is_audible(samples, sample_rate):
            return ""
        if not self.api_key:
            raise VoiceUnavailable("未设置 SPEECH_API_KEY，在线语音识别不可用。")

        wav = encode_wav(samples, sample_rate)
        files = {"file": ("command.wav", wav, "audio/wav")}
        data = {"model": self.model}
        if self.language:
            data["language"] = self.language
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.base_url}/audio/transcriptions"
        try:
            import httpx

            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, headers=headers, data=data, files=files)
                response.raise_for_status()
                payload = response.json()
        except ImportError as exc:
            raise VoiceUnavailable("缺少 httpx，无法调用在线语音识别。") from exc
        except Exception as exc:
            raise VoiceUnavailable(f"在线语音识别失败：{exc}") from exc

        if isinstance(payload, dict):
            text = str(payload.get("text", "")).strip()
        else:
            text = str(payload).strip()
        return text


class SpeechPipeline:
    """Prefer online ASR when configured; fall back to local Whisper."""

    def __init__(
        self,
        local: WhisperRecognizer,
        online: OnlineSpeechRecognizer | None = None,
        *,
        prefer_online: bool = True,
    ) -> None:
        self.local = local
        self.online = online if online is not None else OnlineSpeechRecognizer()
        self.prefer_online = prefer_online

    @property
    def online_ready(self) -> bool:
        return self.prefer_online and self.online.available

    def status(self) -> dict[str, Any]:
        return {
            "online": self.online.status(),
            "online_ready": self.online_ready,
            "local": self.local.status(),
        }

    def transcribe(
        self,
        samples: bytes,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        *,
        require_audible: bool = True,
    ) -> str:
        if not samples:
            return ""
        if require_audible and not is_audible(samples, sample_rate):
            return ""
        if self.online_ready:
            try:
                text = self.online.transcribe(
                    samples, sample_rate, require_audible=False
                )
                if text:
                    return text
                # empty online result: still try local rather than give up
            except VoiceUnavailable:
                # network / auth failure → local fallback
                pass
        return self.local.transcribe(
            samples, sample_rate, require_audible=require_audible
        )


def describe_voice_setup() -> Callable[[], dict[str, Any]]:
    """Factory used by UI to inspect optional voice stack without loading model."""

    def probe() -> dict[str, Any]:
        info: dict[str, Any] = {
            "sounddevice": False,
            "faster_whisper": False,
            "inputs": [],
        }
        try:
            importlib.import_module("sounddevice")
            sounddevice = importlib.import_module("sounddevice")
            info["sounddevice"] = True
            try:
                info["inputs"] = [
                    str(d.get("name", ""))
                    for d in sounddevice.query_devices()
                    if int(d.get("max_input_channels", 0)) > 0
                ]
            except Exception:
                info["inputs"] = []
        except ImportError:
            pass
        try:
            importlib.import_module("faster_whisper")
            info["faster_whisper"] = True
        except ImportError:
            pass
        return info

    return probe
