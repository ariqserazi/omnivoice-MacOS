import torch

from omnivoice.models.omnivoice import VoiceClonePrompt
from omnivoice.utils.voice_profiles import VoiceLibrary


def test_voice_profile_round_trip_and_import_export(tmp_path):
    library = VoiceLibrary(tmp_path / "voices")
    prompt = VoiceClonePrompt(
        ref_audio_tokens=torch.ones(8, 12, dtype=torch.long),
        ref_text="reference transcript.",
        ref_rms=0.08,
        language="English",
        reference_report={"warnings": []},
    )
    waveform = torch.randn(1, 24000)

    profile = library.create_profile(
        name="Narrator A",
        cleaned_audio=waveform,
        sample_rate=24000,
        prompt=prompt,
        transcript="reference transcript.",
        language="English",
        metadata={"reference_report": {"warnings": ["clean clip"]}},
    )

    loaded = library.load_profile(profile.id)
    restored_prompt = library.load_prompt(profile.id)
    assert loaded.display_name == "Narrator A"
    assert restored_prompt.ref_text == "reference transcript."
    assert torch.equal(restored_prompt.ref_audio_tokens, prompt.ref_audio_tokens)

    export_path = library.export_profile(profile.id, tmp_path / "narrator-a.zip")
    imported_library = VoiceLibrary(tmp_path / "imported")
    imported = imported_library.import_profile(export_path)
    assert imported.display_name == "Narrator A"
    assert imported_library.load_profile(imported.id).schema_version == 1
