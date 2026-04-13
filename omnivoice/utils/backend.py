"""Mac-first backend selection and validation helpers."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from omnivoice.utils.app_paths import get_diagnostics_dir

logger = logging.getLogger(__name__)


@dataclass
class BackendResolution:
    requested: str
    selected: str
    fallback_reason: Optional[str]
    dtype: str
    cached_reliability: Optional[bool]
    diagnostics_path: str


@dataclass
class BackendDiagnostics:
    device: str
    reliable: bool
    checked_at: str
    reason: str
    metrics: dict[str, Any]


def available_backends() -> list[str]:
    backends = ["cpu"]
    if torch.backends.mps.is_available():
        backends.insert(0, "mps")
    if torch.cuda.is_available():
        backends.insert(0, "cuda")
    return backends


def get_auto_device() -> str:
    return available_backends()[0]


def normalize_device_name(device: Optional[str]) -> str:
    value = (device or "auto").strip().lower()
    if value in {"auto", ""}:
        return "auto"
    if value.startswith("cuda"):
        return "cuda"
    if value.startswith("mps"):
        return "mps"
    return "cpu"


def recommended_dtype(device: str) -> torch.dtype:
    device = normalize_device_name(device)
    if device == "cuda":
        return torch.float16
    return torch.float32


def dtype_label(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def validate_generated_audio(audio: torch.Tensor, sample_rate: int) -> tuple[bool, dict[str, Any]]:
    """Detect clearly broken outputs so MPS can fall back to CPU."""
    waveform = audio.detach().float().cpu()
    if waveform.dim() == 2:
        waveform = waveform.squeeze(0)
    if waveform.numel() == 0:
        return False, {"reason": "empty", "rms": 0.0, "peak": 0.0}

    finite_ratio = float(torch.isfinite(waveform).float().mean().item())
    if finite_ratio < 1.0:
        return False, {
            "reason": "non_finite",
            "finite_ratio": finite_ratio,
        }

    peak = float(waveform.abs().max().item())
    rms = float(torch.sqrt(torch.mean(torch.square(waveform))).item())
    duration = float(waveform.numel() / max(sample_rate, 1))
    near_zero_ratio = float((waveform.abs() < 1e-4).float().mean().item())

    if duration < 0.15:
        return False, {"reason": "too_short", "duration": duration, "rms": rms, "peak": peak}
    if peak < 1e-3 or rms < 5e-4:
        return False, {"reason": "near_silent", "duration": duration, "rms": rms, "peak": peak}
    if peak > 1.25:
        return False, {"reason": "exploding", "duration": duration, "rms": rms, "peak": peak}
    if near_zero_ratio > 0.995:
        return False, {
            "reason": "mostly_silent",
            "duration": duration,
            "rms": rms,
            "peak": peak,
            "near_zero_ratio": near_zero_ratio,
        }

    return True, {
        "reason": "ok",
        "duration": duration,
        "rms": rms,
        "peak": peak,
        "near_zero_ratio": near_zero_ratio,
    }


def _diagnostics_path(model_id: str) -> Path:
    digest = hashlib.sha1(model_id.encode("utf-8")).hexdigest()[:12]
    return get_diagnostics_dir() / f"{digest}.json"


def load_cached_backend_diagnostics(model_id: str) -> Optional[BackendDiagnostics]:
    path = _diagnostics_path(model_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return BackendDiagnostics(**data)


def save_backend_diagnostics(model_id: str, diagnostics: BackendDiagnostics) -> Path:
    path = _diagnostics_path(model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(diagnostics), indent=2, sort_keys=True))
    return path


def resolve_backend(requested: Optional[str], model_id: str) -> BackendResolution:
    requested_name = normalize_device_name(requested)
    cached = load_cached_backend_diagnostics(model_id)
    fallback_reason = None

    if requested_name == "auto":
        selected = get_auto_device()
        if selected == "mps" and cached and not cached.reliable:
            selected = "cpu"
            fallback_reason = f"cached MPS diagnostic: {cached.reason}"
    else:
        selected = requested_name
        if requested_name == "mps" and not torch.backends.mps.is_available():
            selected = "cpu"
            fallback_reason = "MPS is not available on this machine"
        if requested_name == "cuda" and not torch.cuda.is_available():
            selected = "cpu"
            fallback_reason = "CUDA is not available on this machine"

    if selected not in available_backends():
        selected = "cpu"
        fallback_reason = fallback_reason or "requested backend is unavailable"

    path = _diagnostics_path(model_id)
    return BackendResolution(
        requested=requested_name,
        selected=selected,
        fallback_reason=fallback_reason,
        dtype=dtype_label(recommended_dtype(selected)),
        cached_reliability=None if cached is None else cached.reliable,
        diagnostics_path=str(path),
    )


def describe_backend(resolution: BackendResolution) -> str:
    bits = [f"backend={resolution.selected}", f"dtype={resolution.dtype}"]
    if resolution.requested != "auto":
        bits.insert(0, f"requested={resolution.requested}")
    if resolution.fallback_reason:
        bits.append(f"fallback={resolution.fallback_reason}")
    return ", ".join(bits)
