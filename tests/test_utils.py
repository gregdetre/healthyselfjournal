from pathlib import Path

from healthyselfjournal.utils.time_utils import format_hh_mm_ss, format_mm_ss
from healthyselfjournal.utils.audio_utils import maybe_delete_wav_when_safe


def test_format_mm_ss_and_hh_mm_ss_rounding_and_rollover():
    # mm:ss
    assert format_mm_ss(-3.2) == "0:00"
    assert format_mm_ss(0.4) == "0:00"
    assert format_mm_ss(0.5) == "0:01"
    assert format_mm_ss(59.4) == "0:59"
    assert format_mm_ss(59.5) == "1:00"  # rollover

    # h:mm:ss
    assert format_hh_mm_ss(-1) == "0:00"
    assert format_hh_mm_ss(61) == "1:01"
    assert format_hh_mm_ss(3599.5) == "1:00:00"  # 59:59.5 → 1:00:00
    assert format_hh_mm_ss(3661) == "1:01:01"


def test_maybe_delete_wav_when_safe(tmp_path: Path):
    wav = tmp_path / "seg.wav"
    mp3 = tmp_path / "seg.mp3"
    stt = tmp_path / "seg.stt.json"

    wav.write_bytes(b"RIFF....WAVE")
    # Not safe yet (only wav exists)
    assert maybe_delete_wav_when_safe(wav) is False
    assert wav.exists()

    mp3.write_bytes(b"ID3")
    # Still not safe (stt missing)
    assert maybe_delete_wav_when_safe(wav) is False
    assert wav.exists()

    stt.write_text("{}", encoding="utf-8")
    # Now safe
    assert maybe_delete_wav_when_safe(wav) is True
    assert not wav.exists()
