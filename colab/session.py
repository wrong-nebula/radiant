"""Managed Colab A100 session lifecycle with unconditional EXIT/INT/TERM cleanup."""

from __future__ import annotations

import atexit
import signal
import sys
import time
import uuid
from types import FrameType
from typing import Any

from colab.budget import (
    HARD_CAMPAIGN_CAP_UNITS,
    check_budget_available,
    record_session_usage,
)
from colab.cli_client import ColabClient, SESSION_PREFIX


class ManagedColabSession:
    """Context manager guaranteeing unconditional VM teardown and budget logging."""

    def __init__(
        self,
        *,
        name: str | None = None,
        gpu: str = "A100",
        fallback_gpus: tuple[str, ...] = ("H100", "L4"),
        notes: str = "",
        client: ColabClient | None = None,
        reserve_units: float = 0.5,
        hard_cap: float = HARD_CAMPAIGN_CAP_UNITS,
    ) -> None:
        self.client = client or ColabClient()
        suffix = uuid.uuid4().hex[:6]
        self.name = name or f"{SESSION_PREFIX}{gpu.lower()}-{suffix}"
        if not self.name.startswith(SESSION_PREFIX):
            self.name = f"{SESSION_PREFIX}{self.name}"
        self.gpu = gpu.upper()
        self.fallback_gpus = tuple(g.upper() for g in fallback_gpus if g.upper() != self.gpu)
        self.notes = notes
        self.reserve_units = reserve_units
        self.hard_cap = hard_cap
        self.started_epoch: float | None = None
        self._active = False
        self._cleaned = False
        self._prev_sigint: Any = None
        self._prev_sigterm: Any = None

    def _signal_handler(self, signum: int, _frame: FrameType | None) -> None:
        print(f"\n[colab.session] Caught signal {signum}; tearing down {self.name}...", file=sys.stderr, flush=True)
        self.close(status=f"signal_{signum}")
        sys.exit(128 + signum)

    def __enter__(self) -> "ManagedColabSession":
        remaining = check_budget_available(
            reserve_units=self.reserve_units,
            hard_cap=self.hard_cap,
        )
        print(
            f"[colab.session] Provisioning {self.gpu} session '{self.name}' "
            f"(remaining campaign budget: {remaining:.2f}u / {self.hard_cap:.2f}u)...",
            flush=True,
        )
        self.started_epoch = None
        self._active = True
        atexit.register(self._atexit_cleanup)
        self._prev_sigint = signal.getsignal(signal.SIGINT)
        self._prev_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        last_exc: Exception | None = None
        candidates = [(self.gpu, 2)] + [(fg, 2) for fg in self.fallback_gpus]
        for cand_gpu, tries in candidates:
            for attempt in range(1, tries + 1):
                try:
                    out = self.client.new_session(self.name, gpu=cand_gpu)
                    self.gpu = cand_gpu
                    self.started_epoch = time.time()
                    print(f"[colab.session] Session '{self.name}' ({self.gpu}) ready.\n{out.strip()}", flush=True)
                    return self
                except Exception as exc:
                    last_exc = exc
                    short_err = str(exc).splitlines()[-1] if str(exc) else repr(exc)
                    print(
                        f"[colab.session] Provision attempt {attempt}/{tries} ({cand_gpu}) failed: {short_err}",
                        file=sys.stderr,
                        flush=True,
                    )
                    self.client.stop_session(self.name)
                    time.sleep(10)

        self.close(status="provision_failed")
        assert last_exc is not None
        raise last_exc

    def _atexit_cleanup(self) -> None:
        if self._active and not self._cleaned:
            self.close(status="atexit_cleanup")

    def close(self, status: str = "stopped") -> None:
        if self._cleaned:
            return
        self._cleaned = True
        was_active = self._active
        self._active = False

        if self._prev_sigint is not None:
            try:
                signal.signal(signal.SIGINT, self._prev_sigint)
            except Exception:
                pass
        if self._prev_sigterm is not None:
            try:
                signal.signal(signal.SIGTERM, self._prev_sigterm)
            except Exception:
                pass

        if not was_active:
            return

        print(f"[colab.session] Stopping session '{self.name}' ({status})...", flush=True)
        try:
            self.client.stop_session(self.name)
        except Exception as exc:
            print(f"[colab.session] Warning during stop: {exc}", file=sys.stderr, flush=True)

        # Verify zero managed active sessions remain
        remaining_sessions = self.client.list_managed_active_sessions()
        for orphan in remaining_sessions:
            print(f"[colab.session] Cleaning up leftover managed session '{orphan}'...", flush=True)
            self.client.stop_session(orphan)
        remaining_sessions = self.client.list_managed_active_sessions()
        active_after = len(remaining_sessions)

        now = time.time()
        entry = record_session_usage(
            session_name=self.name,
            hardware=self.gpu,
            started_epoch=self.started_epoch if self.started_epoch is not None else now,
            ended_epoch=now,
            active_sessions_after=active_after,
            status=status,
            notes=self.notes,
        )
        print(
            f"[colab.session] Stopped '{self.name}': {entry.elapsed_seconds:.1f}s, "
            f"{entry.estimated_units:.4f}u (cumulative {entry.cumulative_units:.4f}u), "
            f"active_sessions_after={active_after}",
            flush=True,
        )

    def __exit__(self, exc_type: Any, exc: Any, _tb: Any) -> None:
        status = "completed" if exc_type is None else f"error:{exc_type.__name__}"
        self.close(status=status)

    def exec_code(self, code: str, timeout: int = 3600, stream: bool = True) -> str:
        return self.client.exec_code(self.name, code, timeout=timeout, stream=stream)

    def upload(self, local_path: Any, remote_path: str) -> str:
        return self.client.upload(self.name, local_path, remote_path)

    def download(self, remote_path: str, local_path: Any) -> str:
        return self.client.download(self.name, remote_path, local_path)

    def install(self, packages: list[str], timeout: int = 600) -> str:
        return self.client.install(self.name, packages, timeout=timeout)
