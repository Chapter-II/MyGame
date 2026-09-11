import os
import sys
from types import SimpleNamespace

import pytest

from mygame.input import (
    MicrophoneRecorder,
    OnlineSpeechRecognizer,
    SpeechPipeline,
    VoiceUnavailable,
    WhisperRecognizer,
    is_audible,
)
from mygame.input.voice import _int16_rms


class FakeStream:
    def __init__(self, callback, **kwargs) -> None:
        del kwargs
        self.callback = callback
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True
        self.callback(b"\x01\x00" * 20, 20, None, None)

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.closed = True


def _install_fake_mic(monkeypatch, stream_cls=FakeStream):
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(RawInputStream=lambda **kwargs: stream_cls(**kwargs)),
    )


def test_fake_microphone_lifecycle_and_length_limit(monkeypatch) -> None:
    _install_fake_mic(monkeypatch)
    recorder = MicrophoneRecorder(sample_rate=10, max_seconds=1)
    recorder.start()
    assert recorder.active is True
    samples = recorder.stop()
    assert recorder.active is False
    assert len(samples) == 20
    assert recorder.seconds == 0.0


def test_stop_is_safe_when_not_started() -> None:
    recorder = MicrophoneRecorder()
    assert recorder.stop() == b""
    recorder.abort()  # should not raise


def test_start_twice_does_not_open_second_stream(monkeypatch) -> None:
    created = []

    class TrackingStream(FakeStream):
        def __init__(self, callback, **kwargs) -> None:
            super().__init__(callback, **kwargs)
            created.append(self)

    _install_fake_mic(monkeypatch, TrackingStream)
    recorder = MicrophoneRecorder(sample_rate=10, max_seconds=1)
    recorder.start()
    recorder.start()
    assert len(created) == 1
    recorder.stop()


def test_abort_discards_samples(monkeypatch) -> None:
    _install_fake_mic(monkeypatch)
    recorder = MicrophoneRecorder(sample_rate=10, max_seconds=1)
    recorder.start()
    assert recorder.abort() is None
    assert recorder.active is False


def test_microphone_open_failure_raises_voice_unavailable(monkeypatch) -> None:
    class BrokenStream:
        def __init__(self, **kwargs) -> None:
            del kwargs
            raise OSError("no device")

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(RawInputStream=lambda **kwargs: BrokenStream(**kwargs)),
    )
    recorder = MicrophoneRecorder()
    with pytest.raises(VoiceUnavailable):
        recorder.start()
    assert recorder.active is False


def test_is_audible_rejects_short_and_silent_audio() -> None:
    assert not is_audible(b"")
    assert not is_audible(b"\x00\x00" * 10, sample_rate=16_000)
    loud = (3000).to_bytes(2, "little", signed=True) * (16_000 // 2)
    assert is_audible(loud, sample_rate=16_000)


def test_int16_rms_scale() -> None:
    assert _int16_rms(b"") == 0.0
    full = (32767).to_bytes(2, "little", signed=True) * 100
    assert _int16_rms(full) > 0.9


def test_missing_whisper_model_requires_confirmation(tmp_path) -> None:
    recognizer = WhisperRecognizer(tmp_path / "missing-model", allow_download=False)
    loud = (3000).to_bytes(2, "little", signed=True) * 8000
    with pytest.raises(VoiceUnavailable):
        recognizer.transcribe(loud, sample_rate=16_000)


def test_status_reports_local_model(tmp_path) -> None:
    model_dir = tmp_path / "tiny"
    model_dir.mkdir()
    recognizer = WhisperRecognizer(model_dir, allow_download=False)
    status = recognizer.status()
    assert status["local_path"] == str(model_dir)
    assert status["loaded"] is False
    assert status["allow_download"] is False


def test_transcription_uses_temporary_wav_and_removes_it() -> None:
    captured: list[str] = []

    class Model:
        def transcribe(self, path, **kwargs):
            del kwargs
            captured.append(path)
            return [SimpleNamespace(text=" 前进")], None

    recognizer = WhisperRecognizer("local")
    recognizer._model = Model()
    # skip audible gate by require_audible=False for synthetic bytes
    assert recognizer.transcribe(b"\x00\x00" * 100, require_audible=False) == "前进"
    assert captured
    assert not os.path.exists(captured[0])


def test_transcribe_skips_silence_without_loading_model() -> None:
    class Exploding:
        def transcribe(self, *args, **kwargs):
            raise AssertionError("should not load model for silence")

    recognizer = WhisperRecognizer("local")
    recognizer._model = Exploding()
    assert recognizer.transcribe(b"\x00\x00" * 2000) == ""


def test_empty_samples_returns_empty_without_model() -> None:
    recognizer = WhisperRecognizer("local")
    assert recognizer.transcribe(b"") == ""


def test_encode_wav_header() -> None:
    from mygame.input.voice import encode_wav

    wav = encode_wav(b"\x00\x00" * 1600, sample_rate=16_000)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"


def test_online_recognizer_requires_api_key() -> None:
    online = OnlineSpeechRecognizer(api_key="")
    assert online.available is False
    with pytest.raises(VoiceUnavailable):
        online.transcribe(b"\x01\x00" * 8000, require_audible=False)


def test_online_recognizer_posts_multipart(monkeypatch) -> None:
    from mygame.input import voice as voice_mod

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"text": " 第一侦察队前往北部 "}

    class FakeClient:
        def __init__(self, timeout=None) -> None:
            del timeout
            self.captured = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, data=None, files=None):
            self.captured = {"url": url, "headers": headers, "data": data, "files": files}
            FakeClient.last = self.captured
            return FakeResponse()

    fake_httpx = SimpleNamespace(Client=FakeClient)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    online = voice_mod.OnlineSpeechRecognizer(
        api_key="sk-test",
        base_url="https://asr.example.com/v1",
        model="whisper-1",
        language="zh",
    )
    samples = (2000).to_bytes(2, "little", signed=True) * 8000
    text = online.transcribe(samples, require_audible=False)
    assert text == "第一侦察队前往北部"
    assert FakeClient.last["url"] == "https://asr.example.com/v1/audio/transcriptions"
    assert FakeClient.last["headers"]["Authorization"] == "Bearer sk-test"
    assert "file" in FakeClient.last["files"]


def test_pipeline_prefers_online_then_falls_back() -> None:
    class Local:
        def transcribe(self, samples, sample_rate=16000, **kwargs):
            del samples, sample_rate, kwargs
            return "本地结果"

        def status(self):
            return {}

    class OnlineOK(OnlineSpeechRecognizer):
        def __init__(self):
            super().__init__(api_key="k")

        def transcribe(self, samples, sample_rate=16000, **kwargs):
            del samples, sample_rate, kwargs
            return "在线结果"

    class OnlineFail(OnlineSpeechRecognizer):
        def __init__(self):
            super().__init__(api_key="k")

        def transcribe(self, *args, **kwargs):
            raise VoiceUnavailable("network down")

    samples = (2000).to_bytes(2, "little", signed=True) * 8000

    pipe = SpeechPipeline(Local(), OnlineOK())  # type: ignore[arg-type]
    assert pipe.online_ready is True
    assert pipe.transcribe(samples) == "在线结果"

    pipe2 = SpeechPipeline(Local(), OnlineFail())  # type: ignore[arg-type]
    assert pipe2.transcribe(samples) == "本地结果"

    pipe3 = SpeechPipeline(Local(), OnlineOK(), prefer_online=False)  # type: ignore[arg-type]
    assert pipe3.online_ready is False
    assert pipe3.transcribe(samples) == "本地结果"
