from __future__ import annotations

from healthyselfjournal.utils.pending import (
    count_pending_for_session,
    count_pending_segments,
    iter_pending_segments,
    reconcile_command_for_dir,
    remove_error_sentinel,
    write_error_sentinel,
)


def test_iter_pending_segments_detects_multiple_media_types(tmp_path):
    sessions_dir = tmp_path / "sessions"
    audio_dir = sessions_dir / "250101_1400"
    audio_dir.mkdir(parents=True)

    wav = audio_dir / "clip.wav"
    wav.write_bytes(b"wav")
    webm = audio_dir / "browser-001.webm"
    webm.write_bytes(b"web")
    done = audio_dir / "done.wav"
    done.write_bytes(b"done")
    done.with_suffix(".stt.json").write_text("{}", encoding="utf-8")

    write_error_sentinel(wav, RuntimeError("boom"))

    segments = list(iter_pending_segments(sessions_dir))
    labels = {seg.segment_label: seg for seg in segments}

    assert set(labels) == {"clip.wav", "browser-001.webm"}
    assert labels["clip.wav"].has_error is True
    assert labels["browser-001.webm"].has_error is False
    assert count_pending_segments(sessions_dir) == 2
    assert count_pending_for_session(sessions_dir, "250101_1400") == 2

    command = reconcile_command_for_dir(sessions_dir)
    assert "healthyselfjournal fix stt" in command

    remove_error_sentinel(wav)
    assert not wav.with_suffix(".stt.error.txt").exists()
