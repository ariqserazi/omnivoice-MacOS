import torch

from omnivoice.models.omnivoice import GenerationTask, OmniVoice, OmniVoiceGenerationConfig


class FakeTokenizerConfig:
    frame_rate = 10


class FakeTokenizer:
    config = FakeTokenizerConfig()


class FakeChunkModel:
    def __init__(self):
        self.audio_tokenizer = FakeTokenizer()
        self.calls = []

    def _estimate_target_tokens(self, text, ref_text, num_ref_audio_tokens, speed=1.0):
        return 5

    def _generate_iterative(self, sub_task, gen_config):
        self.calls.append(list(sub_task.ref_audio_tokens))
        return [torch.ones(8, 5, dtype=torch.long) for _ in range(sub_task.batch_size)]


def test_chunked_generation_reuses_reference_prompt(monkeypatch):
    fake_model = FakeChunkModel()
    ref_tokens = torch.arange(16, dtype=torch.long).reshape(8, 2)
    task = GenerationTask(
        batch_size=1,
        texts=["long text"],
        target_lens=[60],
        langs=[None],
        instructs=[None],
        ref_texts=["saved transcript"],
        ref_audio_tokens=[ref_tokens],
        ref_rms=[0.1],
        speed=[1.0],
    )

    monkeypatch.setattr(
        "omnivoice.models.omnivoice.chunk_text_punctuation",
        lambda text, chunk_len, min_chunk_len=3: ["chunk one", "chunk two", "chunk three"],
    )

    results = OmniVoice._generate_chunked(fake_model, task, OmniVoiceGenerationConfig(audio_chunk_duration=5.0))

    assert len(results[0]) == 3
    assert len(fake_model.calls) == 3
    assert all(call[0] is ref_tokens for call in fake_model.calls)
