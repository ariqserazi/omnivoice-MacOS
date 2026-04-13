"""Shared runtime helpers for loading OmniVoice with safer backend handling."""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import torch

from omnivoice import OmniVoice
from omnivoice.utils.backend import (
    BackendDiagnostics,
    BackendResolution,
    describe_backend,
    recommended_dtype,
    resolve_backend,
    save_backend_diagnostics,
    validate_generated_audio,
)

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_backend_sanity_check(model: OmniVoice, device: str) -> BackendDiagnostics:
    """Run a lightweight inference to decide whether the backend is usable."""
    if device != "mps":
        return BackendDiagnostics(
            device=device,
            reliable=True,
            checked_at=_utc_now(),
            reason="sanity check not required",
            metrics={"device": device},
        )

    try:
        audio = model.generate(
            text="Hello from OmniVoice.",
            generation_config=model_generation_config(),
        )[0]
        ok, metrics = validate_generated_audio(audio, model.sampling_rate)
        return BackendDiagnostics(
            device=device,
            reliable=ok,
            checked_at=_utc_now(),
            reason=metrics.get("reason", "unknown"),
            metrics=metrics,
        )
    except Exception as exc:  # pragma: no cover - defensive path
        return BackendDiagnostics(
            device=device,
            reliable=False,
            checked_at=_utc_now(),
            reason=f"exception: {type(exc).__name__}",
            metrics={"error": str(exc)},
        )


def model_generation_config():
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

    return OmniVoiceGenerationConfig(
        num_step=8,
        guidance_scale=1.5,
        postprocess_output=True,
        denoise=True,
    )


def load_model_runtime(
    model_id: str,
    *,
    requested_device: str | None = None,
    load_asr: bool = False,
    asr_model_name: str = "openai/whisper-large-v3-turbo",
    run_backend_check: bool = True,
) -> tuple[OmniVoice, BackendResolution, dict[str, Any]]:
    """Load OmniVoice with backend-aware dtype and optional MPS fallback."""
    resolution = resolve_backend(requested_device, model_id)
    logger.info("Loading OmniVoice with %s", describe_backend(resolution))
    model = OmniVoice.from_pretrained(
        model_id,
        device_map=resolution.selected,
        dtype=recommended_dtype(resolution.selected),
        load_asr=load_asr,
        asr_model_name=asr_model_name,
    )

    diagnostics = None
    if run_backend_check and resolution.selected == "mps":
        diagnostics = run_backend_sanity_check(model, resolution.selected)
        save_backend_diagnostics(model_id, diagnostics)
        if not diagnostics.reliable and normalize_requested_device(requested_device) == "auto":
            logger.warning("MPS backend failed sanity check (%s), falling back to CPU.", diagnostics.reason)
            model = OmniVoice.from_pretrained(
                model_id,
                device_map="cpu",
                dtype=recommended_dtype("cpu"),
                load_asr=load_asr,
                asr_model_name=asr_model_name,
            )
            resolution = resolve_backend("cpu", model_id)

    info = {"resolution": asdict(resolution)}
    if diagnostics is not None:
        info["diagnostics"] = asdict(diagnostics)
    return model, resolution, info


def normalize_requested_device(device: str | None) -> str:
    return "auto" if device is None else device.strip().lower()
