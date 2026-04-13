import torch

from omnivoice.utils.audio import preprocess_reference_audio


def test_preprocess_reference_audio_trims_and_reports():
    sample_rate = 24000
    silence = torch.zeros(1, sample_rate // 2)
    speech = torch.full((1, sample_rate * 2), 0.02)
    waveform = torch.cat([silence, speech, silence], dim=1)

    cleaned, report = preprocess_reference_audio(
        (waveform, sample_rate),
        sample_rate,
        preprocess_prompt=True,
        ref_text="hello there",
    )

    assert cleaned.shape[0] == 1
    assert cleaned.shape[-1] < waveform.shape[-1]
    assert report.cleaned_duration < report.original_duration
    assert any("silence" in op for op in report.operations)
    assert report.rms > 0.02
