from __future__ import annotations

import tempfile
import wave
from pathlib import Path
from typing import Any


class VoiceUnavailable(RuntimeError):
    pass


class MicrophoneRecorder:
    def __init__(self, sample_rate: int = 16_000, max_seconds: int = 15) -> None:
        self.sample_rate = sample_rate
        self.max_samples = sample_rate * max_seconds
        self._chunks: list[bytes] = []
        self._stream: Any = None

    def start(self) -> None:
        try:
            import sounddevice  # type: ignore[import-not-found]
        except ImportError as exc:
            raise VoiceUnavailable("未安装语音依赖，请安装 voice 可选依赖。") from exc
        self._chunks = []

        def callback(indata: Any, frames: int, time: Any, status: Any) -> None:
            del frames, time, status
            if sum(len(chunk) for chunk in self._chunks) < self.max_samples * 2:
                self._chunks.append(bytes(indata))

        self._stream = sounddevice.RawInputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> bytes:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        return b"".join(self._chunks)[: self.max_samples * 2]


class WhisperRecognizer:
    def __init__(self, model: str | Path = "small", allow_download: bool = False) -> None:
        self.model_name = str(model)
        self.allow_download = allow_download
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]
        except ImportError as exc:
            raise VoiceUnavailable("未安装 faster-whisper。") from exc
        is_local = Path(self.model_name).exists()
        if not is_local and not self.allow_download:
            raise VoiceUnavailable("本地语音模型尚未准备；在设置中确认下载后再试。")
        self._model = WhisperModel(
            self.model_name,
            device="cpu",
            compute_type="int8",
            local_files_only=not self.allow_download and not is_local,
        )
        return self._model

    def transcribe(self, samples: bytes, sample_rate: int = 16_000) -> str:
        if not samples:
            return ""
        model = self._load()
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                temporary = handle.name
            with wave.open(temporary, "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(sample_rate)
                output.writeframes(samples)
            segments, _ = model.transcribe(temporary, language="zh", beam_size=3, vad_filter=True)
            return "".join(segment.text for segment in segments).strip()
        finally:
            if temporary:
                Path(temporary).unlink(missing_ok=True)
