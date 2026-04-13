"""Run a lightweight macOS backend diagnostic for OmniVoice."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from omnivoice.utils.runtime import load_model_runtime, run_backend_sanity_check


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose OmniVoice backend reliability on macOS.")
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    model, resolution, runtime_info = load_model_runtime(
        args.model,
        requested_device=args.device,
        run_backend_check=True,
    )
    diagnostics = run_backend_sanity_check(model, resolution.selected)
    print(json.dumps({"runtime": runtime_info, "diagnostics": asdict(diagnostics)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
