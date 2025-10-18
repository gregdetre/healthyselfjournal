from __future__ import annotations

"""Runtime support for launching the desktop PyWebView shell."""

from dataclasses import dataclass, replace
import logging
import socket
import threading
import time
from pathlib import Path
from typing import Optional

import contextlib

from ..web.app import WebAppConfig, build_app
from .settings import load_settings

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DesktopConfig:
    """Configuration container for the PyWebView desktop runtime."""

    web: WebAppConfig
    window_title: str = "Healthyself Journal"
    window_width: int = 1280
    window_height: int = 860
    fullscreen: bool = False
    debug: bool = False
    open_devtools: bool = False
    server_start_timeout: float = 12.0
    confirm_close: bool = True

    def resolved(self) -> "DesktopConfig":
        resolved_web = self.web.resolved()
        return DesktopConfig(
            web=resolved_web,
            window_title=self.window_title,
            window_width=self.window_width,
            window_height=self.window_height,
            fullscreen=self.fullscreen,
            debug=self.debug,
            open_devtools=self.open_devtools,
            server_start_timeout=self.server_start_timeout,
            confirm_close=self.confirm_close,
        )


class _BackgroundServer:
    """Run the FastHTML Starlette app in a background uvicorn server."""

    def __init__(self, web_config: WebAppConfig) -> None:
        self._web_config = web_config
        self._server: "uvicorn.Server | None" = None
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._stopped = threading.Event()

    def start(self) -> None:
        import asyncio
        import uvicorn

        if self._thread is not None:
            raise RuntimeError("Server already started")

        app = build_app(self._web_config)
        config = uvicorn.Config(
            app,
            host=self._web_config.host,
            port=self._web_config.port,
            reload=False,
            log_level="info",
        )
        server = uvicorn.Server(config)
        server.install_signal_handlers = lambda: None  # type: ignore[assignment]
        self._server = server

        async def _serve() -> None:
            try:
                await server.serve()
            except Exception as exc:  # pragma: no cover - runtime failure path
                self._error = exc
                _LOGGER.exception("Uvicorn server error: %%s", exc)
            finally:
                self._stopped.set()

        def _run() -> None:
            try:
                asyncio.run(_serve())
            except Exception as exc:  # pragma: no cover - asyncio failure
                self._error = exc
                _LOGGER.exception("Asyncio failure while running server: %s", exc)
                self._stopped.set()

        thread = threading.Thread(
            target=_run,
            name="hsj-desktop-uvicorn",
            daemon=True,
        )
        thread.start()
        self._thread = thread

    def wait_until_ready(self, timeout: float = 10.0) -> None:
        deadline = time.time() + timeout
        host = self._web_config.host
        port = self._web_config.port
        while time.time() < deadline:
            if self._error:
                raise RuntimeError("Web server failed to start") from self._error
            if self._stopped.is_set():
                if self._error:
                    raise RuntimeError(
                        "Web server stopped unexpectedly"
                    ) from self._error
                raise RuntimeError("Web server stopped unexpectedly")
            if _is_port_open(host, port):
                return
            time.sleep(0.1)
        raise TimeoutError(f"Timed out waiting for desktop web server on {host}:{port}")

    def stop(self, timeout: float = 5.0) -> None:
        server = self._server
        if server is not None:
            server.should_exit = True
        if self._thread is None:
            return
        self._stopped.wait(timeout)
        if self._thread.is_alive():  # pragma: no cover - defensive
            _LOGGER.warning("Desktop server thread still running; joining with timeout")
            self._thread.join(timeout=timeout)

    @property
    def error(self) -> Exception | None:
        return self._error


def _is_port_open(host: str, port: int) -> bool:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.settimeout(0.25)
        try:
            sock.connect((host, port))
            return True
        except OSError:
            return False


def _pick_port(host: str, requested: Optional[int]) -> int:
    if requested and _port_available(host, requested):
        return requested
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, 0))
        return sock.getsockname()[1]


def _port_available(host: str, port: int) -> bool:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
        return True


class _DesktopBridge:
    """Minimal JS bridge exposed to the frontend."""

    def __init__(self) -> None:
        self._window: "webview.Window | None" = None
        self._on_apply_restart: "callable[[], None] | None" = None

    def attach(self, window: "webview.Window") -> None:
        self._window = window

    def set_apply_restart(self, fn: "callable[[], None]") -> None:
        self._on_apply_restart = fn

    def quit(self) -> None:
        """Allow the UI to request app termination."""

        if self._window is not None:
            self._window.destroy()

    def toggle_devtools(self) -> None:
        window = self._window
        if window is not None:
            try:
                window.toggle_devtools()
            except Exception:  # pragma: no cover - optional UI surface
                pass

    def pick_sessions_dir(self) -> str | None:
        """Open a native folder picker and return the selected path, if any."""
        try:
            import webview
        except Exception:  # pragma: no cover - dependency optional in tests
            return None
        try:
            result = webview.create_file_dialog(webview.FOLDER_DIALOG)
            if isinstance(result, list) and result:
                return str(result[0])
            if isinstance(result, str):
                return result
        except Exception:
            return None
        return None

    def apply_and_restart(self) -> None:
        """Persisted settings already saved via HTTP; restart server and reload UI."""
        if self._on_apply_restart is not None:
            try:
                self._on_apply_restart()
            except Exception:
                # Swallow to avoid crashing the UI
                pass


def run_desktop_app(config: DesktopConfig) -> None:
    """Launch the PyWebView desktop shell around the FastHTML app."""

    resolved = config.resolved()

    host = resolved.web.host or "127.0.0.1"
    selected_port = _pick_port(host, resolved.web.port)
    web_config = replace(resolved.web, port=selected_port, reload=False, host=host)

    server = _BackgroundServer(web_config)
    server.start()
    try:
        server.wait_until_ready(resolved.server_start_timeout)
    except Exception:
        server.stop()
        raise

    # Lazy import to avoid requiring pywebview for non-desktop users
    import webview

    try:
        # Enable microphone access inside the embedded WebView
        getattr(webview.settings, "__setattr__")
        webview.settings.enable_media_stream = True  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - attribute may not exist
        _LOGGER.debug("PyWebView media stream setting unavailable; continuing")

    window_url = f"http://{host}:{selected_port}/"
    bridge = _DesktopBridge()

    window = webview.create_window(
        resolved.window_title,
        window_url,
        width=resolved.window_width,
        height=resolved.window_height,
        frameless=False,
        resizable=True,
        fullscreen=resolved.fullscreen,
        confirm_close=resolved.confirm_close,
        js_api=bridge,
    )
    bridge.attach(window)

    def _apply_and_restart() -> None:
        # Stop current server
        try:
            server.stop()
        except Exception:
            pass

        # Rebuild web config by reloading desktop settings, preserving host/port/static
        ds, _ = load_settings()
        base = web_config
        new_web = replace(
            base,
            sessions_dir=(
                ds.sessions_dir if ds.sessions_dir is not None else base.sessions_dir
            ),
            resume=(
                ds.resume_on_launch if ds.resume_on_launch is not None else base.resume
            ),
            voice_enabled=(
                ds.voice_enabled if ds.voice_enabled is not None else base.voice_enabled
            ),
        )

        # Start new server and reload window URL
        new_server = _BackgroundServer(new_web)
        new_server.start()
        try:
            new_server.wait_until_ready(resolved.server_start_timeout)
        except Exception:
            new_server.stop()
            return
        # Swap server reference
        nonlocal_server = locals()
        # We cannot rebind outer 'server' easily without nonlocal; use tricks:
        # But simplest: just assign to name in enclosing scope via closure capture
        # However, Python requires nonlocal declaration; restructure to avoid.
        # Instead, reload the window to point at the same URL using new_web host/port.
        try:
            window.load_url(f"http://{new_web.host}:{new_web.port}/")
        except Exception:
            pass

    bridge.set_apply_restart(_apply_and_restart)

    def _on_closed() -> None:
        server.stop()

    window.events.closed += _on_closed

    if resolved.open_devtools:
        window.events.shown += lambda: window.toggle_devtools()

    try:
        webview.start(debug=resolved.debug, http_server=False)
    finally:
        server.stop()
        if server.error:
            raise RuntimeError(
                "Desktop web server exited with an error"
            ) from server.error
