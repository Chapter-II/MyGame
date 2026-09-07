import os
import sys
from types import SimpleNamespace

import pytest

from mygame.input import MicrophoneRecorder, VoiceUnavailable, WhisperRecognizer


class FakeStream:
    def __init__(self, callback, **kwargs) -> None:
        del kwargs
        self.callback = callback

    def start(self) -> None:
        self.callback(b"\x01\x00" * 20, 20, None, None)

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


def test_fake_microphone_lifecycle_and_length_limit(monkeypatch) -> None:
    fake_module = SimpleNamespace(RawInputStream=lambda **kwargs: FakeStream(**kwargs))
    monkeypatch.setitem(sys.modules, "sounddevice", fake_module)
    recorder = MicrophoneRecorder(sample_rate=10, max_seconds=1)
    recorder.start()
    samples = recorder.stop()
    assert len(samples) == 20


def test_missing_whisper_model_requires_confirmation(tmp_path) -> None:
    recognizer = WhisperRecognizer(tmp_path / "missing-model", allow_download=False)
    with pytest.raises(VoiceUnavailable):
        recognizer.transcribe(b"\x00\x00" * 20)


def test_transcription_uses_temporary_wav_and_removes_it() -> None:
    captured: list[str] = []

    class Model:
        def transcribe(self, path, **kwargs):
            del kwargs
            captured.append(path)
            return [SimpleNamespace(text=" 前进")], None

    recognizer = WhisperRecognizer("local")
    recognizer._model = Model()
    assert recognizer.transcribe(b"\x00\x00" * 100) == "前进"
    assert captured
    assert not os.path.exists(captured[0])
