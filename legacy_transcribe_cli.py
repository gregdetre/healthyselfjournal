import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
import importlib
import os

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
console = Console()


def _run_ffmpeg_record(output_wav: Path, duration: Optional[float]) -> None:
    # Use avfoundation on macOS: ":0" is default input device (system mic)
    # -ac 1 mono, -ar 16000 sample rate for Whisper
    # If duration is provided, use -t; otherwise record until stdin closes.
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "avfoundation",
        "-i",
        ":0",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-y",
    ]
    if duration is not None and duration > 0:
        cmd += ["-t", str(duration)]
    # Write PCM S16LE WAV
    cmd += [str(output_wav)]

    if duration is None:
        # Start ffmpeg and keep stdin open; pressing RETURN will terminate by sending EOF
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            console.print("[bold cyan]Recording...[/] press RETURN to stop")
            sys.stdin.readline()
        finally:
            # Terminate politely, then kill if needed
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
    else:
        subprocess.run(cmd, check=True)


def _transcribe_with_mlx(
    audio_source: Path, model_name: str, language: Optional[str]
) -> None:
    model_id = (
        model_name if "/" in model_name else f"mlx-community/whisper-{model_name}"
    )
    cmd = [
        "mlx_whisper",
        "--model",
        model_id,
        "--task",
        "transcribe",
        str(audio_source),
    ]
    if language:
        cmd[1:1] = ["--language", language]
    # Stream output so user sees progress and the transcript
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env={**os.environ, "HF_HUB_ENABLE_TELEMETRY": "0"},
    )
    output_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        output_lines.append(line)
        if line:
            console.print(line)
    ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd, "\n".join(output_lines))


def _transcribe_with_faster(
    audio_source: Path,
    model_name: str,
    compute_type: str,
    cpu_threads: Optional[int],
    language: Optional[str],
) -> None:
    try:
        faster_whisper = importlib.import_module("faster_whisper")
    except ModuleNotFoundError:
        console.print("Error: faster-whisper not installed. Run: uv add faster-whisper")
        raise typer.Exit(code=5)

    WhisperModel = getattr(faster_whisper, "WhisperModel")
    model = WhisperModel(
        model_name,
        device="cpu",
        compute_type=compute_type,
        cpu_threads=cpu_threads if cpu_threads is not None else 0,
    )
    segments, _info = model.transcribe(str(audio_source), language=language)
    text = "".join(segment.text for segment in segments).strip()
    console.print(text)


@app.callback(invoke_without_command=True)
def record(
    duration: Optional[float] = typer.Option(
        None, help="Record only this many seconds; omit to record until RETURN"
    ),
    backend: str = typer.Option(
        "mlx",
        help="Backend: mlx (GPU, default), faster (CPU), auto (try MLX then faster)",
        rich_help_panel="Transcription",
    ),
    input_path: Optional[Path] = typer.Option(
        None,
        "--input",
        "-i",
        help="Path to audio/video file to transcribe. If provided, skip recording",
        rich_help_panel="Input",
    ),
    model_name: str = typer.Option(
        "large-v2",
        help="Model to use (e.g., tiny, base, small, medium, large-v2)",
        rich_help_panel="Transcription",
    ),
    compute_type: str = typer.Option(
        "int8",
        help="Compute precision: int8, int8_float16, float16, float32",
        rich_help_panel="Transcription",
    ),
    cpu_threads: Optional[int] = typer.Option(
        None,
        help="CPU threads for faster-whisper (default: library decides)",
        rich_help_panel="Transcription",
    ),
    language: Optional[str] = typer.Option(
        "en",
        help="Language code (e.g., en, fr). Default: en",
        rich_help_panel="Transcription",
    ),
):
    """Record from mic or transcribe a file. Defaults to MLX (GPU) + large-v2 with language 'en'."""

    # Determine audio source
    if input_path is not None:
        if not input_path.exists():
            console.print(f"[red]Error:[/] file not found: {input_path}")
            raise typer.Exit(code=1)
        audio_source = input_path
        console.print(
            f"[bold green]Transcribing file.[/] Using model [bold]{model_name}[/]..."
        )
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "mic.wav"
            if duration is not None and duration > 0:
                console.print(f"[bold cyan]Recording for {duration} seconds...[/]")
            _run_ffmpeg_record(wav_path, duration)
            console.print(
                f"[bold green]Finished recording.[/] Transcribing with [bold]{model_name}[/]..."
            )
            audio_source = wav_path

            # Transcribe within tempdir context
            if backend in ("auto", "mlx"):
                try:
                    _transcribe_with_mlx(audio_source, model_name, language)
                    return
                except FileNotFoundError:
                    if backend == "mlx":
                        console.print(
                            "Error: mlx_whisper CLI not found. Install with: uv add mlx-whisper"
                        )
                        raise typer.Exit(code=6)
                    # auto fallback to faster
                except subprocess.CalledProcessError as e:
                    if backend == "mlx":
                        console.print(
                            f"[red]MLX transcription failed:[/] {str(e).strip()}"
                        )
                        raise typer.Exit(code=7)
                    # auto fallback to faster
            # Faster-whisper path
            _transcribe_with_faster(
                audio_source, model_name, compute_type, cpu_threads, language
            )
            return

    # If we reach here, we're in file-input mode (no tempdir context)
    if backend in ("auto", "mlx"):
        try:
            _transcribe_with_mlx(audio_source, model_name, language)
            return
        except FileNotFoundError:
            if backend == "mlx":
                console.print(
                    "Error: mlx_whisper CLI not found. Install with: uv add mlx-whisper"
                )
                raise typer.Exit(code=6)
            # auto fallback to faster
        except subprocess.CalledProcessError as e:
            if backend == "mlx":
                console.print(f"[red]MLX transcription failed:[/] {str(e).strip()}")
                raise typer.Exit(code=7)
            # auto fallback to faster

    _transcribe_with_faster(
        audio_source, model_name, compute_type, cpu_threads, language
    )
    return


if __name__ == "__main__":
    app()
