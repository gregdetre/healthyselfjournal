from __future__ import annotations

"""Persistent model management utilities for local AI backends."""

from dataclasses import dataclass, field
import os
import hashlib
import json
import platform
import shutil
import time
from pathlib import Path
from typing import Any, Dict

import httpx
from platformdirs import user_data_dir
from tqdm import tqdm

_METADATA_VERSION = 1


@dataclass(slots=True)
class _ModelMetadata:
    files: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)


class ModelManager:
    """Manage lifecycle of local AI model assets under platformdirs storage."""

    def __init__(self) -> None:
        root = Path(user_data_dir("HealthySelfJournal", "Experim"))
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)
        self._models_dir = self._root / "models"
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_path = self._root / "model_metadata.json"
        self._metadata: Dict[str, Any] = self._load_metadata()

    def ensure_faster_whisper_model(self, model_name: str) -> Path:
        """Return download root for faster-whisper and validate existing cache."""

        root = self._models_dir / "faster-whisper"
        root.mkdir(parents=True, exist_ok=True)
        model_dir = root / model_name

        metadata = self._metadata.setdefault("faster-whisper", {}).get(model_name)

        if model_dir.exists():
            if metadata:
                if not self._verify_directory(model_dir, metadata):
                    shutil.rmtree(model_dir, ignore_errors=True)
                    self._metadata["faster-whisper"].pop(model_name, None)
                    self._write_metadata()
            else:
                # No metadata recorded yet; capture fresh values for existing cache
                self.record_faster_whisper_model(model_name)

        return root

    def record_faster_whisper_model(self, model_name: str) -> None:
        """Persist file hashes for a downloaded faster-whisper model."""

        root = self._models_dir / "faster-whisper"
        model_dir = root / model_name
        if not model_dir.exists():
            return

        files: Dict[str, Dict[str, Any]] = {}
        for path in model_dir.rglob("*"):
            if path.is_file():
                rel = str(path.relative_to(model_dir))
                files[rel] = {
                    "sha256": self._hash_file(path),
                    "size": path.stat().st_size,
                }

        meta_root = self._metadata.setdefault("faster-whisper", {})
        meta_root[model_name] = {
            "version": _METADATA_VERSION,
            "files": files,
            "updated_at": time.time(),
        }
        self._write_metadata()

    def validate_faster_whisper_model(self, model_name: str) -> bool:
        """Return True if cached files match recorded metadata."""

        root = self._models_dir / "faster-whisper"
        model_dir = root / model_name
        metadata = self._metadata.get("faster-whisper", {}).get(model_name)
        if not model_dir.exists() or not metadata:
            return False
        return self._verify_directory(model_dir, metadata)

    def suggest_faster_whisper_device(self) -> str:
        """Suggest optimal device for faster-whisper on this machine."""

        system = platform.system().lower()
        machine = platform.machine().lower()
        if system == "darwin" and machine in {"arm64", "aarch64"}:
            return "metal"
        return "cpu"

    def faster_whisper_model_dir(self, model_name: str) -> Path:
        return self._models_dir / "faster-whisper" / model_name

    def ensure_llama_model(
        self, model_name: str, *, url: str | None = None, sha256: str | None = None
    ) -> Path:
        """Ensure a local llama model exists; optionally download/verify."""

        models_root = self._models_dir / "llama"
        models_root.mkdir(parents=True, exist_ok=True)
        target = models_root / model_name

        metadata_root = self._metadata.setdefault("llama", {})
        metadata = metadata_root.get(model_name)

        if target.exists():
            if metadata:
                if not self._verify_directory(target.parent, metadata):
                    target.unlink(missing_ok=True)
                    metadata_root.pop(model_name, None)
                    self._write_metadata()
            elif sha256:
                if not self._verify_file_hash(target, sha256):
                    target.unlink(missing_ok=True)
            else:
                self.record_llama_model(model_name, target)
                return target

        if not target.exists():
            if not url:
                raise FileNotFoundError(
                    "Local LLM model not found. Provide a download URL or place the model file manually."
                )
            self._download_file(url, target, sha256=sha256)
            self.record_llama_model(model_name, target)
        else:
            if sha256 and not self._verify_file_hash(target, sha256):
                raise RuntimeError(
                    "Local LLM model checksum mismatch after verification."
                )
            if not metadata:
                self.record_llama_model(model_name, target)

        return target

    def record_llama_model(self, model_name: str, file_path: Path) -> None:
        """Store checksum metadata for a llama gguf file."""

        if not file_path.exists():
            return

        meta_root = self._metadata.setdefault("llama", {})
        meta_root[model_name] = {
            "version": _METADATA_VERSION,
            "files": {
                file_path.name: {
                    "sha256": self._hash_file(file_path),
                    "size": file_path.stat().st_size,
                }
            },
            "updated_at": time.time(),
        }
        self._write_metadata()

    def llama_model_path(self, model_name: str) -> Path:
        return self._models_dir / "llama" / model_name

    def delete_llama_model(self, model_name: str) -> bool:
        """Delete a managed llama gguf file and remove metadata.

        Returns True if a file or metadata entry was removed; False otherwise.
        """

        models_root = self._models_dir / "llama"
        target = models_root / model_name
        removed_any = False

        try:
            if target.exists():
                target.unlink(missing_ok=True)
                removed_any = True
        except Exception:
            # Continue to metadata cleanup even if file removal fails
            pass

        meta_root = self._metadata.setdefault("llama", {})
        if meta_root.pop(model_name, None) is not None:
            removed_any = True
            self._write_metadata()

        return removed_any

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_metadata(self) -> Dict[str, Any]:
        try:
            if self._metadata_path.exists():
                raw = self._metadata_path.read_text(encoding="utf-8")
                data = json.loads(raw)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {"version": _METADATA_VERSION}

    def _write_metadata(self) -> None:
        try:
            payload = json.dumps(self._metadata, indent=2, sort_keys=True)
            self._metadata_path.write_text(payload, encoding="utf-8")
        except Exception:
            pass

    def _verify_directory(self, directory: Path, metadata: Dict[str, Any]) -> bool:
        files_meta = metadata.get("files") or {}
        for rel, info in files_meta.items():
            file_path = directory / rel
            try:
                if not file_path.exists():
                    return False
                expected = info.get("sha256")
                if expected and expected != self._hash_file(file_path):
                    return False
            except Exception:
                return False
        return True

    def _verify_file_hash(self, path: Path, expected: str) -> bool:
        try:
            calculated = self._hash_file(path)
        except Exception:
            return False
        return calculated.lower() == expected.lower()

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _download_file(
        self, url: str, destination: Path, *, sha256: str | None = None
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        headers: Dict[str, str] = {}
        # Support private/gated Hugging Face downloads when a token is provided
        token = (
            os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_TOKEN")
        )
        if token:
            headers["Authorization"] = f"Bearer {token}"
        with httpx.stream(
            "GET", url, headers=headers, follow_redirects=True, timeout=120.0
        ) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length") or 0)
            with (
                destination.open("wb") as fh,
                tqdm(
                    total=total if total > 0 else None,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=destination.name,
                    leave=True,
                ) as bar,
            ):
                for chunk in response.iter_bytes(1024 * 128):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    try:
                        bar.update(len(chunk))
                    except Exception:
                        pass
        if sha256 and not self._verify_file_hash(destination, sha256):
            destination.unlink(missing_ok=True)
            raise ValueError(
                "Downloaded model failed checksum validation; file removed."
            )


_manager: ModelManager | None = None


def get_model_manager() -> ModelManager:
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager
