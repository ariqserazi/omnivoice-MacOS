"""Voice profile management CLI for OmniVoice."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict

from omnivoice.models.omnivoice import OmniVoiceGenerationConfig
from omnivoice.utils.audio import save_audio_file_any
from omnivoice.utils.runtime import load_model_runtime, run_backend_sanity_check
from omnivoice.utils.voice_profiles import VoiceLibrary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage reusable OmniVoice voice profiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--voices-dir", type=str, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Create a saved voice profile.")
    create.add_argument("--name", required=True)
    create.add_argument("--ref-audio", required=True)
    create.add_argument("--ref-text", default=None)
    create.add_argument("--language", default=None)
    create.add_argument("--notes", default="")
    create.add_argument("--tags", nargs="*", default=[])
    create.add_argument("--model", default="k2-fsa/OmniVoice")
    create.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    create.add_argument("--no-asr", action="store_true")
    create.add_argument("--skip-backend-check", action="store_true")

    list_cmd = subparsers.add_parser("list", help="List saved voices.")
    list_cmd.add_argument("--json", action="store_true")

    generate = subparsers.add_parser("generate", help="Generate audio from a saved voice.")
    generate.add_argument("--voice", required=True)
    generate.add_argument("--text", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--model", default="k2-fsa/OmniVoice")
    generate.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    generate.add_argument("--language", default=None)
    generate.add_argument("--num-step", type=int, default=32)
    generate.add_argument("--guidance-scale", type=float, default=2.0)
    generate.add_argument("--speed", type=float, default=1.0)
    generate.add_argument("--duration", type=float, default=None)
    generate.add_argument("--skip-backend-check", action="store_true")

    rename = subparsers.add_parser("rename", help="Rename a saved voice.")
    rename.add_argument("--voice", required=True)
    rename.add_argument("--new-name", required=True)

    delete = subparsers.add_parser("delete", help="Delete a saved voice.")
    delete.add_argument("--voice", required=True)

    export = subparsers.add_parser("export", help="Export a saved voice as a zip archive.")
    export.add_argument("--voice", required=True)
    export.add_argument("--output", required=True)

    import_cmd = subparsers.add_parser("import", help="Import a saved voice from a zip archive.")
    import_cmd.add_argument("--archive", required=True)
    import_cmd.add_argument("--replace", action="store_true")

    diagnose = subparsers.add_parser("diagnose-mac", help="Run the MPS backend diagnostic.")
    diagnose.add_argument("--model", default="k2-fsa/OmniVoice")
    diagnose.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])

    return parser


def _print_profiles(library: VoiceLibrary, as_json: bool) -> None:
    profiles = library.list_profiles()
    if as_json:
        print(json.dumps([asdict(profile) for profile in profiles], indent=2))
        return
    if not profiles:
        print("No saved voices found.")
        return
    for profile in profiles:
        print(
            f"{profile.display_name}\t{profile.id}\t{profile.duration_seconds:.1f}s\t"
            f"{profile.updated_at}"
        )


def main(argv=None) -> int:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        force=True,
    )
    args = build_parser().parse_args(argv)
    library = VoiceLibrary(args.voices_dir)

    if args.command == "list":
        _print_profiles(library, args.json)
        return 0

    if args.command == "rename":
        profile = library.rename_profile(args.voice, args.new_name)
        print(f"Renamed voice profile to {profile.display_name} ({profile.id})")
        return 0

    if args.command == "delete":
        profile = library.delete_profile(args.voice)
        print(f"Deleted voice profile {profile.display_name} ({profile.id})")
        return 0

    if args.command == "export":
        path = library.export_profile(args.voice, args.output)
        print(str(path))
        return 0

    if args.command == "import":
        profile = library.import_profile(args.archive, replace=args.replace)
        print(f"Imported voice profile {profile.display_name} ({profile.id})")
        return 0

    if args.command == "diagnose-mac":
        model, resolution, runtime_info = load_model_runtime(
            args.model,
            requested_device=args.device,
            run_backend_check=True,
        )
        diagnostics = run_backend_sanity_check(model, resolution.selected)
        print(json.dumps({"runtime": runtime_info, "diagnostics": asdict(diagnostics)}, indent=2))
        return 0

    if args.command == "create":
        model, _, _ = load_model_runtime(
            args.model,
            requested_device=args.device,
            load_asr=not args.no_asr,
            run_backend_check=not args.skip_backend_check,
        )
        prepared = model.prepare_voice_clone_reference(
            args.ref_audio,
            ref_text=args.ref_text,
            language=args.language,
            preprocess_prompt=True,
        )
        prompt = model.create_voice_clone_prompt(
            (prepared["waveform"], model.sampling_rate),
            ref_text=prepared["ref_text"],
            language=args.language,
            preprocess_prompt=False,
        )
        profile = library.create_profile(
            name=args.name,
            cleaned_audio=prepared["waveform"],
            sample_rate=model.sampling_rate,
            prompt=prompt,
            transcript=prepared["ref_text"],
            language=args.language,
            notes=args.notes,
            tags=args.tags,
            metadata={"reference_report": prepared["report"].to_dict()},
        )
        print(json.dumps(asdict(profile), indent=2))
        return 0

    if args.command == "generate":
        model, _, _ = load_model_runtime(
            args.model,
            requested_device=args.device,
            run_backend_check=not args.skip_backend_check,
        )
        prompt = library.load_prompt(args.voice)
        config = OmniVoiceGenerationConfig(
            num_step=args.num_step,
            guidance_scale=args.guidance_scale,
        )
        audio = model.generate(
            text=args.text,
            language=args.language,
            voice_clone_prompt=prompt,
            speed=args.speed,
            duration=args.duration,
            generation_config=config,
        )[0]
        save_audio_file_any(args.output, audio, model.sampling_rate)
        print(args.output)
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
