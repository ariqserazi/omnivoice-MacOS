from pathlib import Path

import torch

from omnivoice.utils.backend import BackendDiagnostics, resolve_backend, validate_generated_audio


def test_validate_generated_audio_rejects_non_finite():
    waveform = torch.tensor([[0.0, float("nan"), 0.2]])
    ok, metrics = validate_generated_audio(waveform, 24000)
    assert not ok
    assert metrics["reason"] == "non_finite"


def test_resolve_backend_falls_back_from_cached_unreliable_mps(monkeypatch, tmp_path):
    monkeypatch.setenv("OMNIVOICE_APP_HOME", str(tmp_path))
    monkeypatch.setattr("omnivoice.utils.backend.available_backends", lambda: ["mps", "cpu"])
    monkeypatch.setattr("omnivoice.utils.backend.get_auto_device", lambda: "mps")
    monkeypatch.setattr(
        "omnivoice.utils.backend.load_cached_backend_diagnostics",
        lambda model_id: BackendDiagnostics(
            device="mps",
            reliable=False,
            checked_at="2026-01-01T00:00:00+00:00",
            reason="near_silent",
            metrics={},
        ),
    )

    resolution = resolve_backend("auto", "fake-model")

    assert resolution.selected == "cpu"
    assert "cached MPS diagnostic" in resolution.fallback_reason
