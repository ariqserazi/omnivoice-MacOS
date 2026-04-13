# OmniVoice for macOS

Mac-first local voice cloning built on OmniVoice, with reusable saved voice profiles, a Gradio app, CLI tools, safer Apple Silicon defaults, and a workflow aimed at normal local use instead of one-off research demos.

This repo extends the existing OmniVoice codepaths rather than replacing them. The main additions are:

- reusable local voice profiles
- saved conditioning reuse for later generations
- reference audio cleanup and warnings
- CPU-first startup for stability on Apple Silicon
- MPS diagnostics and fallback paths
- a friendlier Gradio workflow for cloning, saving, and reusing voices

## What You Can Do

- Clone a voice from a short reference clip
- Save that cloned voice locally as a named profile
- Reuse saved voices later without reuploading the reference audio
- Rename, delete, export, and import voice profiles
- Generate from raw reference audio or from a saved profile
- Run a macOS backend diagnostic

## Current Recommendation

For this repo on Apple Silicon, use **CPU by default** unless you specifically want to experiment with MPS.

Why:

- MPS can work, but it is more likely to hit memory pressure on long generations
- CPU is slower, but much more stable for day-to-day local use
- the launcher in this repo now defaults to CPU for that reason

## Quick Start

From the repo root:

```bash
./run_omnivoice_demo.sh
```

That launcher:

- uses your existing `.venv` install if available
- does not reinstall everything on every run
- defaults to `cpu`
- prints the local URL clearly before launch

Open:

```text
http://127.0.0.1:7860
```

If you want to force a reinstall or resync:

```bash
OMNIVOICE_SYNC=1 ./run_omnivoice_demo.sh
```

If you want to try MPS manually:

```bash
OMNIVOICE_DEVICE=mps ./run_omnivoice_demo.sh
```

## Manual Setup

### venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
./run_omnivoice_demo.sh
```

### uv

If you prefer `uv`:

```bash
uv sync --extra dev
uv run omnivoice-demo --device cpu --ip 127.0.0.1 --port 7860
```

## Gradio App

The local app is launched with:

```bash
omnivoice-demo --device cpu --ip 127.0.0.1 --port 7860
```

The app is organized into four sections:

- `Voice Clone`
- `Saved Voices`
- `Voice Design`
- `Diagnostics / Settings`

Default user flow:

1. Upload a short clean reference clip
2. Analyze it
3. Review or edit the transcript
4. Save it as a voice profile
5. Reuse that saved voice later for new text

## Saved Voice Profiles

Saved voices live here on macOS by default:

```text
~/Library/Application Support/OmniVoice/voices
```

Override that with:

- `--voices-dir /custom/path`
- `OMNIVOICE_VOICES_DIR=/custom/path`

Each voice profile stores:

- a unique id
- display name
- timestamps
- cleaned reference audio
- approved transcript
- language if provided
- duration
- metadata and schema version
- cached reusable voice conditioning data derived from the real model prompt path

This is an honest representation, not a fake speaker preset. The app reuses the actual saved voice clone prompt data where available.

## CLI

### Launch the app

```bash
omnivoice-demo --device cpu --ip 127.0.0.1 --port 7860
```

### Create a saved voice

```bash
omnivoice-voices create \
  --name "Narrator A" \
  --ref-audio ref.wav \
  --ref-text "Reference transcript"
```

### List saved voices

```bash
omnivoice-voices list
```

### Generate from a saved voice

```bash
omnivoice-voices generate \
  --voice "Narrator A" \
  --text "Welcome back." \
  --output welcome.wav
```

### Use saved voice directly from the single-item inference CLI

```bash
omnivoice-infer \
  --voice "Narrator A" \
  --text "Hello there." \
  --output out.wav
```

### Rename, export, import, delete

```bash
omnivoice-voices rename --voice "Narrator A" --new-name "Narrator Warm"
omnivoice-voices export --voice "Narrator Warm" --output narrator-warm.zip
omnivoice-voices import --archive narrator-warm.zip
omnivoice-voices delete --voice "Narrator Warm"
```

### Diagnose macOS backend behavior

```bash
omnivoice-diagnose-mac --device auto
```

## Python Example

```python
from omnivoice import OmniVoice
import soundfile as sf
import torch

model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cpu",
    dtype=torch.float32,
)

audio = model.generate(
    text="Hello, this is a local voice cloning test.",
    ref_audio="ref.wav",
    ref_text="Reference transcript.",
)

sf.write("out.wav", audio[0].squeeze(0).cpu().numpy(), 24000)
```

## Best Reference Audio

For better clone quality and consistency:

- use a clean 3-10 second clip
- prefer one speaker only
- avoid background music
- avoid room echo
- avoid clipped recordings
- avoid long dead air
- keep the transcript accurate

Bad transcript alignment can reduce voice similarity even if the audio itself is good.

## Backend Notes

This repo includes Mac-specific backend logic:

- safer dtype selection on macOS
- backend diagnostics
- MPS validation
- fallback handling when MPS behaves badly

In practice, CPU is still the recommended default on many Macs for stability.

If you want to experiment:

- `--device cpu` for stability
- `--device mps` for Apple GPU testing
- `--device auto` to let the app choose

## TorchAudio / TorchCodec Note

This repo has been patched to avoid depending on `torchcodec` for the common audio load/save paths used by the app and CLI. Audio I/O now prefers `soundfile` and `pydub` where appropriate.

## Files Most Relevant To This Mac Workflow

- [pyproject.toml](pyproject.toml)
- [run_omnivoice_demo.sh](run_omnivoice_demo.sh)
- [omnivoice/cli/demo.py](omnivoice/cli/demo.py)
- [omnivoice/cli/infer.py](omnivoice/cli/infer.py)
- [omnivoice/cli/voices.py](omnivoice/cli/voices.py)
- [omnivoice/cli/diagnose_mac.py](omnivoice/cli/diagnose_mac.py)
- [omnivoice/models/omnivoice.py](omnivoice/models/omnivoice.py)
- [omnivoice/utils/audio.py](omnivoice/utils/audio.py)
- [omnivoice/utils/backend.py](omnivoice/utils/backend.py)
- [omnivoice/utils/runtime.py](omnivoice/utils/runtime.py)
- [omnivoice/utils/voice_profiles.py](omnivoice/utils/voice_profiles.py)
- [docs/mac-voice-workflow.md](docs/mac-voice-workflow.md)

## Limitations

- Saved voice profiles improve reuse and consistency, but they do not remove the base model’s inherent speaker drift limits.
- Long-form generation can still vary across chunks.
- Poor source audio cannot be fully fixed by preprocessing.
- CPU is more stable on many Macs, but it is slower than MPS when MPS behaves well.

## Training, Evaluation, and Research Material

The original OmniVoice project also includes training and evaluation code. See:

- [examples/](examples/)
- [docs/training.md](docs/training.md)
- [docs/evaluation.md](docs/evaluation.md)
- [docs/voice-design.md](docs/voice-design.md)
- [docs/languages.md](docs/languages.md)

## Upstream Model

Base model and paper:

- Hugging Face: [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice)
- Paper: [arXiv:2604.00688](https://arxiv.org/abs/2604.00688)

## Citation

```bibtex
@article{zhu2026omnivoice,
      title={OmniVoice: Towards Omnilingual Zero-Shot Text-to-Speech with Diffusion Language Models},
      author={Zhu, Han and Ye, Lingxuan and Kang, Wei and Yao, Zengwei and Guo, Liyong and Kuang, Fangjun and Han, Zhifeng and Zhuang, Weiji and Lin, Long and Povey, Daniel},
      journal={arXiv preprint arXiv:2604.00688},
      year={2026}
}
```
