# OmniVoice 🌍

<p align="center">
  <img width="200" height="200" alt="OmniVoice" src="https://zhu-han.github.io/omnivoice/pics/omnivoice.jpg" />
</p>

<p align="center">
  <a href="https://huggingface.co/k2-fsa/OmniVoice"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E" alt="Hugging Face Model"></a>
  &nbsp;
  <a href="https://huggingface.co/spaces/k2-fsa/OmniVoice"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue" alt="Hugging Face Space"></a>
  &nbsp;
  <a href="https://arxiv.org/abs/2604.00688"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a>
  &nbsp;
  <a href="https://zhu-han.github.io/omnivoice"><img src="https://img.shields.io/badge/GitHub.io-Demo_Page-blue?logo=GitHub&style=flat-square"></a>
</p>

OmniVoice is a state-of-the-art massively multilingual zero-shot text-to-speech (TTS) model supporting over 600 languages. Built on a novel diffusion language model-style architecture, it generates high-quality speech with superior inference speed, supporting voice cloning and voice design.

This repository also includes a **Mac-first local voice cloning workflow** with:

- saved reusable voice profiles
- reference-audio preprocessing and warnings
- MPS vs CPU backend diagnostics and fallback
- a Gradio app tuned for repeated everyday use on Apple Silicon

**Contents**: [Key Features](#key-features) | [Installation](#installation) | [Quick Start](#quick-start) | [Python API](#python-api) | [Command-Line Tools](#command-line-tools) | [Training & Evaluation](#training--evaluation) | [Discussion](#discussion--communication) | [Citation](#citation)

## Key Features

- **600+ Languages Supported**: The broadest language coverage among zero-shot TTS models ([full list](docs/languages.md)).
- **Voice Cloning**: State-of-the-art voice cloning quality.
- **Voice Design**: Control voices via assigned speaker attributes (gender, age, pitch, dialect/accent, whisper, etc.).
- **Fine-grained Control**: Non-verbal symbols (e.g., `[laughter]`) and pronunciation correction via pinyin or phonemes.
- **Fast Inference**: RTF as low as 0.025 (40x faster than real-time).
- **Diffusion Language Model-style Architecture**: A clean, streamlined, and scalable design that delivers both quality and speed.

---

## Installation

Choose **one** of the following methods: **pip** or **uv**.

### pip

> We recommend using a fresh virtual environment (e.g., `conda`, `venv`, etc.) to avoid conflicts.

**Step 1**: Install PyTorch

<details>
<summary>NVIDIA GPU</summary>

```bash
# Install pytorch with your CUDA version, e.g.
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
```
> See [PyTorch official site](https://pytorch.org/get-started/locally/) for other versions installation.

</details>

<details>
<summary>Apple Silicon</summary>

```bash
pip install torch==2.8.0 torchaudio==2.8.0
```

For the improved local workflow in this repo, we recommend:

```bash
pip install -e ".[dev]"
```

</details>

**Step 2**: Install OmniVoice (choose one)

```bash
# From PyPI (stable release)
pip install omnivoice

# From the latest source on GitHub (no need to clone)
pip install git+https://github.com/k2-fsa/OmniVoice.git

# For development (clone first, editable install)
git clone https://github.com/k2-fsa/OmniVoice.git
cd OmniVoice
pip install -e .
```

### uv

Clone the repository and sync dependencies:

```bash
git clone https://github.com/k2-fsa/OmniVoice.git
cd OmniVoice
uv sync
```

> **Tip**: Can use mirror with `uv sync --default-index "https://mirrors.aliyun.com/pypi/simple"`

---

## Quick Start

Try OmniVoice without coding:

- Launch the local web UI: `omnivoice-demo --device auto --ip 0.0.0.0 --port 8001`

- Or use the one-command local launcher from the repo root:

```bash
./run_omnivoice_demo.sh
```

- Create a reusable saved voice from the CLI:

```bash
omnivoice-voices create \
  --name "Narrator A" \
  --ref-audio ref.wav \
  --ref-text "Reference transcript"
```

- Reuse a saved voice later without re-uploading the clip:

```bash
omnivoice-infer \
  --voice "Narrator A" \
  --text "Welcome back." \
  --output out.wav
```


- Or try it directly on [HuggingFace Space](https://huggingface.co/spaces/k2-fsa/OmniVoice)

> If you have trouble connecting to HuggingFace when downloading the pre-trained models, set `export HF_ENDPOINT="https://hf-mirror.com"` before running.

For full usage, see the [Python API](#python-api) and [Command-Line Tools](#command-line-tools) sections below.

For the full macOS workflow, backend troubleshooting, and saved-voice details, see [docs/mac-voice-workflow.md](docs/mac-voice-workflow.md).

---

## Python API

OmniVoice supports three generation modes. All features in this section are also available via [command-line tools](#command-line-tools).

### Voice Cloning

Clone a voice from a short reference audio. Provide `ref_audio` and `ref_text`:

```python
from omnivoice import OmniVoice
import torch
import soundfile as sf

model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0",
    dtype=torch.float16
)
# Apple Silicon users: use device_map="mps" instead

audio = model.generate(
    text="Hello, this is a test of zero-shot voice cloning.",
    ref_audio="ref.wav",
    ref_text="Transcription of the reference audio.",
) # audio is a list of `torch.Tensor` with shape (1, T) at 24 kHz.

# If you don't want to input `ref_text` manually, you can directly omit the `ref_text`.
# The model will use Whisper ASR to auto-transcribe it.

sf.write("out.wav", audio[0].squeeze(0).cpu().numpy(), 24000)
```

> **Tips**
>
> - Use a clean **3-10 second** single-speaker reference clip.
> - Avoid background music, room echo, clipped recordings, and long dead air.
> - An accurate transcript matters. Mismatched transcripts can reduce voice similarity and stability.
> - For better results with Arabic numerals, normalize them to words first (e.g., "123" → "one hundred twenty-three") with text normalization tools (e.g., [WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing)).

### Voice Design

Describe the desired voice with speaker attributes — no reference audio needed.
Supported attributes: **gender** (male/female), **age** (child to elderly),
**pitch** (very low to very high), **style** (whisper), **English accent**
(American, British, etc.), and **Chinese dialect** (四川话, 陕西话, etc.).
Attributes are comma-separated and freely combinable across categories.

```python
audio = model.generate(
    text="Hello, this is a test of zero-shot voice design.",
    instruct="female, low pitch, british accent",
)
```

See [docs/voice-design.md](docs/voice-design.md) for the full attribute
reference, Chinese equivalents, and usage tips.

### Auto Voice

Let the model choose a voice automatically:

```python
audio = model.generate(text="This is a sentence without any voice prompt.")
```

### Generation Parameters

All above three modes share the same `model.generate()` API. You can further control the generation behavior via keyword arguments:

```python
audio = model.generate(
    text="...",
    num_step=32,  # diffusion steps (or 16 for faster inference)
    speed=1.0,     # speed factor (>1.0 faster, <1.0 slower)
    duration=10.0, # fixed output duration in seconds (overrides speed)
    # ... more options
)
```
See more detailed control in [docs/generation-parameters.md](docs/generation-parameters.md).

### Non-Verbal & Pronunciation Control

OmniVoice supports inline **non-verbal symbols** and **pronunciation correction** within the input text.

**Non-verbal symbols**: Insert tags like `[laughter]` directly in the text to add expressive non-verbal sounds.

```python
audio = model.generate(text="[laughter] You really got me. I didn't see that coming at all.")
```

Supported tags: `[laughter]`, `[sigh]`, `[confirmation-en]`, `[question-en]`, `[question-ah]`, `[question-oh]`, `[question-ei]`, `[question-yi]`, `[surprise-ah]`, `[surprise-oh]`, `[surprise-wa]`, `[surprise-yo]`, `[dissatisfaction-hnn]`.

**Pronunciation control (Chinese)**: Use pinyin with tone numbers to correct specific character pronunciations.

```python
audio = model.generate(text="这批货物打ZHE2出售后他严重SHE2本了，再也经不起ZHE1腾了。")
```

**Pronunciation control (English)**: Use [CMU pronunciation dictionary](https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict.0.7a)  (uppercase, in brackets) to override default English pronunciations.

```python
audio = model.generate(text="He plays the [B EY1 S] guitar while catching a [B AE1 S] fish.")
```

---

## Command-Line Tools

Five CLI entry points are provided. The CLI tools support all features available in the Python API plus saved voice profile workflows and Mac backend diagnostics.

| Command | Description | Source |
|---|---|---|
| `omnivoice-demo` | Interactive Gradio web demo | [omnivoice/cli/demo.py](omnivoice/cli/demo.py) |
| `omnivoice-infer` | Single-item inference | [omnivoice/cli/infer.py](omnivoice/cli/infer.py) |
| `omnivoice-infer-batch` | Batch inference across multiple GPUs | [omnivoice/cli/infer_batch.py](omnivoice/cli/infer_batch.py) |
| `omnivoice-voices` | Create, list, rename, delete, export, and import saved voice profiles | [omnivoice/cli/voices.py](omnivoice/cli/voices.py) |
| `omnivoice-diagnose-mac` | Run the Apple Silicon backend sanity check | [omnivoice/cli/diagnose_mac.py](omnivoice/cli/diagnose_mac.py) |

### Demo

```bash
omnivoice-demo --device auto --ip 0.0.0.0 --port 8001
```

Provides a web UI with four sections:

- `Voice Clone`: analyze a reference clip, edit the transcript, save a reusable voice profile, or generate a one-off sample
- `Saved Voices`: browse local voices, preview metadata, rename/delete, export/import, and generate from a saved profile
- `Voice Design`: create prompt-based voices without a reference clip
- `Diagnostics / Settings`: inspect the active backend, switch MPS vs CPU, and run a quick benchmark

Saved voices default to `~/Library/Application Support/OmniVoice/voices` on macOS. Override this with `--voices-dir` or `OMNIVOICE_VOICES_DIR`.

### Single Inference

```bash
# Voice Cloning
# ref_text can be omitted (Whisper will auto-transcribe ref_audio to get it).
omnivoice-infer \
    --model k2-fsa/OmniVoice \
    --text "This is a test for text to speech." \
    --ref_audio ref.wav \
    --ref_text "Transcription of the reference audio." \
    --output hello.wav

# Voice Design
omnivoice-infer --model k2-fsa/OmniVoice \
    --text "This is a test for text to speech." \
    --instruct "male, British accent" \
    --output hello.wav

# Auto Voice
omnivoice-infer \
    --model k2-fsa/OmniVoice \
    --text "This is a test for text to speech."\
    --output hello.wav

# Reuse a saved voice profile
omnivoice-infer \
    --voice "Narrator A" \
    --text "This uses the saved voice profile." \
    --device auto \
    --output narrator.wav
```

`omnivoice-infer --device auto` prefers CUDA, then MPS, then CPU. On macOS it uses `float32` for MPS and CPU, runs an MPS sanity check, and falls back to CPU automatically if MPS output looks broken, silent, or non-finite.

### Saved Voice Profiles

```bash
# Create and cache a reusable voice profile
omnivoice-voices create \
    --name "Narrator A" \
    --ref-audio ref.wav \
    --ref-text "Reference transcript"

# List saved voices
omnivoice-voices list

# Generate from a saved voice
omnivoice-voices generate \
    --voice "Narrator A" \
    --text "Welcome back." \
    --output welcome.wav

# Rename, export, import, delete
omnivoice-voices rename --voice "Narrator A" --new-name "Narrator Warm"
omnivoice-voices export --voice "Narrator Warm" --output narrator-warm.zip
omnivoice-voices import --archive narrator-warm.zip
omnivoice-voices delete --voice "Narrator Warm"
```

Each saved voice profile stores:

- cleaned reference audio
- approved transcript
- cached reusable `VoiceClonePrompt` conditioning tokens
- reference RMS and metadata
- schema and app version metadata

### Apple Silicon Diagnostics

```bash
omnivoice-diagnose-mac --device auto
```

Use this if MPS sounds noisy, silent, or unstable. The diagnostic result is cached locally and `--device auto` will prefer CPU automatically when a previous MPS check failed.

### Limitations

- Saved voice profiles reuse the model's real voice clone prompt representation, but they do not magically bypass the model's intrinsic speaker drift limits.
- Very long passages can still vary slightly across chunks, although the same saved conditioning is now reused across chunked generation.
- Poor reference audio or inaccurate transcripts will still hurt quality even with preprocessing and warnings.

### Batch Inference

`omnivoice-infer-batch` can distribute batch inference across multiple GPUs, designed for large-scale TTS tasks.

```bash
omnivoice-infer-batch \
    --model k2-fsa/OmniVoice \
    --test_list test.jsonl \
    --res_dir results/
```

The test list is a JSONL file where each line is a JSON object:
```json
{"id": "sample_001", "text": "Hello world", "ref_audio": "/path/to/ref.wav", "ref_text": "Reference transcript", "instruct": "female, british accent", "language_id": "en", "language_name": "English", "duration": 10.0, "speed": 1.0}
```
Only `id` and `text` are mandatory fields. `ref_audio` and `ref_text` are used in voice cloning mode. `instruct` is used in voice design mode. If no reference audio or instruct are provided, the model will generate text in a random voice.

`language_id`, `language_name`, `duration`, and `speed` are optional. `duration` (in seconds) fixes the output length; `speed` controls the speaking rate. If `duration` and `speed` are both provided, `speed` will be ignored.

---

## Training & Evaluation

See [examples/](examples/) for the complete pipeline — from data preparation to training, evaluation, and finetuning.

---

## Discussion & Communication

You can directly discuss on [GitHub Issues](https://github.com/k2-fsa/OmniVoice/issues).

You can also scan the QR code to join our wechat group or follow our wechat official account.

| Wechat Group | Wechat Official Account |
| ------------ | ----------------------- |
|![wechat](https://k2-fsa.org/zh-CN/assets/pic/wechat_group.jpg) |![wechat](https://k2-fsa.org/zh-CN/assets/pic/wechat_account.jpg) |

---

## Citation

```bibtex
@article{zhu2026omnivoice,
      title={OmniVoice: Towards Omnilingual Zero-Shot Text-to-Speech with Diffusion Language Models},
      author={Zhu, Han and Ye, Lingxuan and Kang, Wei and Yao, Zengwei and Guo, Liyong and Kuang, Fangjun and Han, Zhifeng and Zhuang, Weiji and Lin, Long and Povey, Daniel},
      journal={arXiv preprint arXiv:2604.00688},
      year={2026}
}
```
