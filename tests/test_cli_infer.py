import torch

from omnivoice.models.omnivoice import VoiceClonePrompt
from omnivoice.utils.voice_profiles import VoiceLibrary


class FakeModel:
    sampling_rate = 24000

    def __init__(self):
        self.last_generate_kwargs = None

    def generate(self, **kwargs):
        self.last_generate_kwargs = kwargs
        return [torch.zeros(1, 2400)]


def test_infer_uses_saved_voice_profile(monkeypatch, tmp_path):
    from omnivoice.cli import infer

    library = VoiceLibrary(tmp_path / "voices")
    prompt = VoiceClonePrompt(
        ref_audio_tokens=torch.ones(8, 10, dtype=torch.long),
        ref_text="saved transcript.",
        ref_rms=0.05,
    )
    library.create_profile(
        name="Saved Voice",
        cleaned_audio=torch.zeros(1, 2400),
        sample_rate=24000,
        prompt=prompt,
        transcript="saved transcript.",
    )

    fake_model = FakeModel()
    monkeypatch.setattr(
        infer,
        "load_model_runtime",
        lambda *args, **kwargs: (
            fake_model,
            type("Res", (), {"selected": "cpu"})(),
            {"resolution": {"selected": "cpu"}},
        ),
    )
    saved_paths = []
    monkeypatch.setattr("omnivoice.cli.infer.torchaudio.save", lambda path, audio, sr: saved_paths.append((path, sr)))

    infer.main(
        [
            "--text",
            "Hello there",
            "--output",
            str(tmp_path / "out.wav"),
            "--voice",
            "Saved Voice",
            "--voices-dir",
            str(tmp_path / "voices"),
        ]
    )

    assert fake_model.last_generate_kwargs["voice_clone_prompt"].ref_text == "saved transcript."
    assert saved_paths == [(str(tmp_path / "out.wav"), 24000)]
