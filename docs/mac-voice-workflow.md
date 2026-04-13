# Mac Voice Workflow

This repository is tuned for a practical local macOS voice-cloning workflow, especially on Apple Silicon.

## Recommended Reference Audio

- Use a clean 3-10 second clip.
- Prefer one speaker only.
- Avoid background music, crowd noise, room echo, and phone speaker playback.
- Avoid clipped recordings and long leading or trailing silence.
- Keep the transcript accurate. Wrong transcripts reduce clone quality.

## Saved Voice Profiles

Saved voice profiles are stored locally at:

```text
~/Library/Application Support/OmniVoice/voices
```

Override the directory with:

- CLI: `--voices-dir /custom/path`
- Environment variable: `OMNIVOICE_VOICES_DIR=/custom/path`

Each profile contains:

- `profile.json`: metadata, transcript, language, timestamps, schema version
- `conditioning.pt`: cached reusable `VoiceClonePrompt` data
- `reference.wav`: cleaned managed copy of the reference audio

The saved representation is honest and model-backed:

- cached reference audio tokens
- transcript used for conditioning
- reference RMS for volume matching
- cleaned reference audio for preview/export/import

## Backend Selection on Mac

`--device auto` behaves like this:

1. Prefer MPS when available.
2. Use `float32` on MPS and CPU for safety and consistency.
3. Run a lightweight sanity check on MPS.
4. Fall back to CPU if the result is silent, non-finite, or clearly broken.

Diagnostic metadata is cached under:

```text
~/Library/Application Support/OmniVoice/diagnostics
```

## Useful Commands

Create a saved voice:

```bash
omnivoice-voices create \
  --name "Narrator A" \
  --ref-audio ref.wav \
  --ref-text "Reference transcript"
```

Generate from that saved voice:

```bash
omnivoice-infer \
  --voice "Narrator A" \
  --text "Hello from a saved voice." \
  --output out.wav
```

Run the backend diagnostic:

```bash
omnivoice-diagnose-mac --device auto
```

## Troubleshooting

If MPS output sounds noisy, broken, or empty:

- run `omnivoice-diagnose-mac --device auto`
- retry with `--device cpu`
- relaunch `omnivoice-demo` and switch the backend in `Diagnostics / Settings`

If voice similarity is weak:

- shorten the reference clip to 3-10 seconds
- remove noisy or echoed clips
- fix the transcript before saving the profile
- reuse the saved profile instead of repeatedly uploading slightly different clips

## Current Limits

- This improves reuse and consistency, but it does not change the base model architecture.
- Long-form synthesis can still drift slightly across chunks.
- Very poor source audio cannot be repaired by preprocessing alone.
