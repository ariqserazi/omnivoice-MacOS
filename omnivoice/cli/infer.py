"""Single-item inference CLI for OmniVoice."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict

import torchaudio

from omnivoice.models.omnivoice import OmniVoiceGenerationConfig
from omnivoice.utils.common import str2bool
from omnivoice.utils.runtime import load_model_runtime
from omnivoice.utils.voice_profiles import VoiceLibrary


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OmniVoice single-item inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="k2-fsa/OmniVoice")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize.")
    parser.add_argument("--output", type=str, required=True, help="Output WAV file path.")
    parser.add_argument("--ref_audio", type=str, default=None)
    parser.add_argument("--ref_text", type=str, default=None)
    parser.add_argument(
        "--voice",
        "--voice-profile",
        dest="voice",
        type=str,
        default=None,
        help="Saved voice profile name or id to reuse without re-uploading reference audio.",
    )
    parser.add_argument("--instruct", type=str, default=None)
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--num_step", type=int, default=32)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--t_shift", type=float, default=0.1)
    parser.add_argument("--denoise", type=str2bool, default=True)
    parser.add_argument("--postprocess_output", type=str2bool, default=True)
    parser.add_argument("--preprocess_prompt", type=str2bool, default=True)
    parser.add_argument("--layer_penalty_factor", type=float, default=5.0)
    parser.add_argument("--position_temperature", type=float, default=5.0)
    parser.add_argument("--class_temperature", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cpu", "cuda"],
        help="Backend override. Auto prefers CUDA, then MPS, then CPU.",
    )
    parser.add_argument(
        "--skip-backend-check",
        action="store_true",
        help="Skip the first-run MPS sanity check.",
    )
    parser.add_argument(
        "--voices-dir",
        type=str,
        default=None,
        help="Override the default local voice profile library path.",
    )
    parser.add_argument(
        "--print-runtime",
        action="store_true",
        help="Print runtime backend and diagnostics metadata as JSON.",
    )
    return parser


def main(argv=None):
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        force=True,
    )
    args = get_parser().parse_args(argv)

    library = VoiceLibrary(args.voices_dir)
    model, resolution, runtime_info = load_model_runtime(
        args.model,
        requested_device=args.device,
        run_backend_check=not args.skip_backend_check,
    )
    logging.info("Runtime: %s", json.dumps(runtime_info, indent=2))

    generation_config = OmniVoiceGenerationConfig(
        num_step=args.num_step,
        guidance_scale=args.guidance_scale,
        t_shift=args.t_shift,
        denoise=args.denoise,
        preprocess_prompt=args.preprocess_prompt,
        postprocess_output=args.postprocess_output,
        layer_penalty_factor=args.layer_penalty_factor,
        position_temperature=args.position_temperature,
        class_temperature=args.class_temperature,
    )

    generate_kwargs = dict(
        text=args.text,
        language=args.language,
        instruct=args.instruct,
        duration=args.duration,
        speed=args.speed,
        generation_config=generation_config,
    )

    if args.voice:
        profile = library.find_profile(args.voice)
        generate_kwargs["voice_clone_prompt"] = library.load_prompt(profile.id)
        logging.info("Using saved voice profile: %s (%s)", profile.display_name, profile.id)
    else:
        generate_kwargs["ref_audio"] = args.ref_audio
        generate_kwargs["ref_text"] = args.ref_text

    logging.info("Generating audio for: %s...", args.text[:80])
    audios = model.generate(**generate_kwargs)
    torchaudio.save(args.output, audios[0], model.sampling_rate)
    logging.info("Saved to %s", args.output)

    if args.print_runtime:
        print(json.dumps({"runtime": runtime_info, "resolution": asdict(resolution)}, indent=2))


if __name__ == "__main__":
    main()
