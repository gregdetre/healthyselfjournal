from pathlib import Path

from healthyselfjournal.utils.time_utils import format_hh_mm_ss, format_mm_ss
from healthyselfjournal.utils.audio_utils import (
    maybe_delete_wav_when_safe,
    extension_for_media_type,
    should_discard_short_answer,
)
from healthyselfjournal.utils.session_layout import (
    next_cli_segment_name,
    next_web_segment_name,
    build_segment_path,
)
from healthyselfjournal.utils.session_utils import get_max_recorded_index


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


def test_extension_for_media_type_known_values():
    assert extension_for_media_type("audio/webm", "clip.webm") == ".webm"
    assert extension_for_media_type("audio/ogg;codecs=opus", "clip.bin") == ".ogg"
    assert extension_for_media_type("audio/mpeg", None) == ".mp3"
    assert extension_for_media_type("application/octet-stream", "clip.wav") == ".wav"


def test_should_discard_short_answer_helper():
    assert should_discard_short_answer(0.5, 0.2)
    assert not should_discard_short_answer(5.0, 3.0)


def test_session_layout_helpers(tmp_path: Path):
    audio_dir = tmp_path / "sess"
    audio_dir.mkdir()
    (audio_dir / "session123_01.wav").write_bytes(b"data")
    idx, basename = next_cli_segment_name("session123", audio_dir, start_index=1)
    assert idx == 2
    assert basename == "session123_02"
    path = build_segment_path(audio_dir, basename, ".wav")
    assert path.name == "session123_02.wav"

    (audio_dir / "browser-001.webm").write_bytes(b"data")
    (audio_dir / "browser-002.ogg").write_bytes(b"data")
    widx, wname = next_web_segment_name(audio_dir, start_index=1)
    assert widx == 3
    assert wname == "browser-003"


def test_get_max_recorded_index_with_mixed_segments(tmp_path: Path):
    audio_dir = tmp_path / "sess"
    audio_dir.mkdir()
    (audio_dir / "session999_03.wav").write_bytes(b"data")
    (audio_dir / "browser-004.webm").write_bytes(b"data")
    assert get_max_recorded_index(audio_dir, "session999") == 4
