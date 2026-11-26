"""Thin, reliable subprocess client around google-colab-cli."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

COLAB_BIN = Path(shutil.which("colab") or (Path.home() / ".local" / "bin" / "colab"))
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "colab" / ".colab_sessions.json"
SESSION_PREFIX = "radiant-gi-"


class ColabCLIError(RuntimeError):
    """Raised when a google-colab-cli invocation fails."""


class ColabClient:
    def __init__(
        self,
        colab_bin: Path = COLAB_BIN,
        config_path: Path = DEFAULT_CONFIG_PATH,
        auth: str = "oauth2",
    ) -> None:
        self.colab_bin = Path(colab_bin)
        self.config_path = Path(config_path)
        self.auth = auth
        if not self.colab_bin.exists():
            raise FileNotFoundError(f"google-colab-cli not found at {self.colab_bin}")

    def _base_cmd(self) -> list[str]:
        return [
            str(self.colab_bin),
            "--config",
            str(self.config_path),
            "--auth",
            self.auth,
        ]

    def _run(
        self,
        args: Sequence[str],
        *,
        input_text: str | None = None,
        timeout: int | None = 600,
        stream: bool = False,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        cmd = self._base_cmd() + list(args)
        if stream:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE if input_text is not None else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            if input_text is not None and proc.stdin is not None:
                proc.stdin.write(input_text)
                proc.stdin.close()
            out_lines: list[str] = []
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                out_lines.append(line)
            ret = proc.wait(timeout=timeout)
            output = "".join(out_lines)
            if check and ret != 0:
                raise ColabCLIError(f"Command failed ({ret}): {' '.join(cmd)}\n{output[-2000:]}")
            return subprocess.CompletedProcess(cmd, ret, output, "")

        res = subprocess.run(
            cmd,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if check and res.returncode != 0:
            raise ColabCLIError(
                f"Command failed ({res.returncode}): {' '.join(cmd)}\n"
                f"STDOUT:\n{res.stdout[-2000:]}\nSTDERR:\n{res.stderr[-2000:]}"
            )
        return res

    def new_session(self, session_name: str, gpu: str = "A100") -> str:
        args = ["new", "-s", session_name]
        if gpu and gpu.upper() != "CPU":
            args += ["--gpu", gpu.upper()]
        res = self._run(args, timeout=300)
        return res.stdout + res.stderr

    def stop_session(self, session_name: str) -> str:
        res = self._run(["stop", "-s", session_name], timeout=120, check=False)
        return res.stdout + res.stderr

    def install(self, session_name: str, packages: Sequence[str], timeout: int = 600) -> str:
        res = self._run(
            ["install", "-s", session_name, *packages],
            timeout=timeout,
            stream=True,
        )
        return res.stdout

    def exec_code(
        self,
        session_name: str,
        code: str,
        *,
        timeout: int = 3600,
        stream: bool = True,
        retries: int = 4,
    ) -> str:
        import time as _time
        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                res = self._run(
                    ["exec", "-s", session_name, "--timeout", str(timeout)],
                    input_text=code,
                    timeout=timeout + 60,
                    stream=stream,
                )
                return res.stdout
            except ColabCLIError as exc:
                last_exc = exc
                msg = str(exc)
                if attempt < retries and ("Connection was lost" in msg or "Timeout" in msg or "503" in msg or "502" in msg):
                    print(f"[colab.cli] Transient exec error (attempt {attempt}/{retries}); retrying in 5s...", flush=True)
                    _time.sleep(5)
                    continue
                raise
        assert last_exc is not None
        raise last_exc

    def exec_file(
        self,
        session_name: str,
        script_path: Path,
        *,
        timeout: int = 3600,
        stream: bool = True,
    ) -> str:
        code = Path(script_path).read_text(encoding="utf-8")
        return self.exec_code(session_name, code, timeout=timeout, stream=stream)

    def upload(self, session_name: str, local_path: Path, remote_path: str, retries: int = 4) -> str:
        import time as _time
        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                res = self._run(
                    ["upload", "-s", session_name, str(local_path), remote_path],
                    timeout=600,
                )
                return res.stdout + res.stderr
            except ColabCLIError as exc:
                last_exc = exc
                if attempt < retries:
                    print(f"[colab.cli] Upload retry {attempt}/{retries} for {remote_path}...", flush=True)
                    _time.sleep(5)
                    continue
                raise
        assert last_exc is not None
        raise last_exc

    def download(self, session_name: str, remote_path: str, local_path: Path) -> str:
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        res = self._run(
            ["download", "-s", session_name, remote_path, str(local_path)],
            timeout=600,
        )
        return res.stdout + res.stderr

    def list_sessions_raw(self) -> str:
        res = self._run(["sessions"], timeout=60, check=False)
        return res.stdout

    def list_managed_active_sessions(self, prefix: str = SESSION_PREFIX) -> list[str]:
        """Return active sessions owned by this harness (matching prefix or local config)."""
        found: set[str] = set()
        raw = self.list_sessions_raw()
        for line in raw.splitlines():
            m = re.match(r"^\s*\[([^\]]+)\]", line)
            if m:
                name = m.group(1).strip()
                if name != "?" and name.startswith(prefix):
                    found.add(name)
        if self.config_path.exists():
            try:
                cfg = json.loads(self.config_path.read_text(encoding="utf-8"))
                sessions_dict = cfg.get("sessions", cfg) if isinstance(cfg, dict) else {}
                if isinstance(sessions_dict, dict):
                    for k in sessions_dict:
                        if k.startswith(prefix) and k in raw:
                            found.add(k)
            except Exception:
                pass
        return sorted(found)

    def usage(self) -> str:
        res = self._run(["usage"], timeout=60, check=False)
        return res.stdout
