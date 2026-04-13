#!/usr/bin/env python3
"""Gradio demo for OmniVoice with saved local voice profiles."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from typing import Any

import gradio as gr
import numpy as np

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.backend import available_backends
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name
from omnivoice.utils.runtime import load_model_runtime
from omnivoice.utils.voice_profiles import VoiceLibrary

logger = logging.getLogger(__name__)

_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)

_CATEGORIES = {
    "Gender / 性别": ["Male / 男", "Female / 女"],
    "Age / 年龄": [
        "Child / 儿童",
        "Teenager / 少年",
        "Young Adult / 青年",
        "Middle-aged / 中年",
        "Elderly / 老年",
    ],
    "Pitch / 音调": [
        "Very Low Pitch / 极低音调",
        "Low Pitch / 低音调",
        "Moderate Pitch / 中音调",
        "High Pitch / 高音调",
        "Very High Pitch / 极高音调",
    ],
    "Style / 风格": ["Whisper / 耳语"],
    "English Accent / 英文口音": [
        "American Accent / 美式口音",
        "Australian Accent / 澳大利亚口音",
        "British Accent / 英国口音",
        "Chinese Accent / 中国口音",
        "Canadian Accent / 加拿大口音",
        "Indian Accent / 印度口音",
        "Korean Accent / 韩国口音",
        "Portuguese Accent / 葡萄牙口音",
        "Russian Accent / 俄罗斯口音",
        "Japanese Accent / 日本口音",
    ],
    "Chinese Dialect / 中文方言": [
        "Henan Dialect / 河南话",
        "Shaanxi Dialect / 陕西话",
        "Sichuan Dialect / 四川话",
        "Guizhou Dialect / 贵州话",
        "Yunnan Dialect / 云南话",
        "Guilin Dialect / 桂林话",
        "Jinan Dialect / 济南话",
        "Shijiazhuang Dialect / 石家庄话",
        "Gansu Dialect / 甘肃话",
        "Ningxia Dialect / 宁夏话",
        "Qingdao Dialect / 青岛话",
        "Northeast Dialect / 东北话",
    ],
}

_ATTR_INFO = {
    "English Accent / 英文口音": "Only effective for English speech.",
    "Chinese Dialect / 中文方言": "Only effective for Chinese speech.",
}


class AppRuntime:
    def __init__(self, checkpoint: str, device: str, load_asr: bool, voices_dir: str | None):
        self.checkpoint = checkpoint
        self.load_asr = load_asr
        self.voice_library = VoiceLibrary(voices_dir)
        self.model: OmniVoice | None = None
        self.runtime_info: dict[str, Any] = {}
        self.selected_backend = device
        self.reload(device, load_asr=load_asr)

    def reload(self, device: str, load_asr: bool | None = None) -> dict[str, Any]:
        if load_asr is not None:
            self.load_asr = load_asr
        self.model, resolution, runtime_info = load_model_runtime(
            self.checkpoint,
            requested_device=device,
            load_asr=self.load_asr,
            run_backend_check=True,
        )
        self.runtime_info = runtime_info
        self.selected_backend = resolution.selected
        return runtime_info

    def summarize_runtime(self) -> str:
        resolution = self.runtime_info.get("resolution", {})
        diagnostics = self.runtime_info.get("diagnostics")
        lines = [
            f"**Active backend:** `{resolution.get('selected', 'unknown')}`",
            f"**Requested backend:** `{resolution.get('requested', 'auto')}`",
            f"**Dtype:** `{resolution.get('dtype', 'unknown')}`",
            f"**Saved voices:** `{len(self.voice_library.list_profiles())}`",
            f"**Voice library:** `{self.voice_library.root_dir}`",
        ]
        if resolution.get("fallback_reason"):
            lines.append(f"**Fallback:** {resolution['fallback_reason']}")
        if diagnostics:
            lines.append(
                f"**Last backend check:** `{diagnostics.get('reason', 'unknown')}` at `{diagnostics.get('checked_at', 'n/a')}`"
            )
        return "\n\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnivoice-demo",
        description="Launch a Gradio demo for OmniVoice.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--root-path", default=None)
    parser.add_argument("--share", action="store_true", default=False)
    parser.add_argument("--no-asr", action="store_true", default=False)
    parser.add_argument("--voices-dir", default=None)
    return parser


def _lang_dropdown(label="Language (optional) / 语种 (可选)", value="Auto"):
    return gr.Dropdown(
        label=label,
        choices=_ALL_LANGUAGES,
        value=value,
        allow_custom_value=False,
        interactive=True,
        info="Keep as Auto to auto-detect the language.",
    )


def _normalize_language(language: str | None) -> str | None:
    if not language or language == "Auto":
        return None
    return language


def _wave_to_numpy(audio, sample_rate: int):
    waveform = audio.squeeze(0).detach().cpu().numpy()
    return sample_rate, (waveform * 32767).astype(np.int16)


def _voice_choices(library: VoiceLibrary):
    profiles = library.list_profiles()
    return [(f"{profile.display_name} ({profile.id[:8]})", profile.id) for profile in profiles]


def _format_profile(profile) -> str:
    report = (profile.metadata or {}).get("reference_report", {})
    warnings = report.get("warnings", [])
    return "\n\n".join(
        [
            f"**Name:** {profile.display_name}",
            f"**ID:** `{profile.id}`",
            f"**Created:** `{profile.created_at}`",
            f"**Duration:** `{profile.duration_seconds:.2f}s`",
            f"**Language:** `{profile.language or 'Auto'}`",
            f"**Transcript:** {profile.transcript}",
            f"**Warnings:** {'; '.join(warnings) if warnings else 'None'}",
        ]
    )


def _format_reference_report(report_dict: dict[str, Any]) -> str:
    warnings = report_dict.get("warnings") or []
    operations = report_dict.get("operations") or []
    return "\n\n".join(
        [
            f"**Cleaned duration:** `{report_dict.get('cleaned_duration', 0.0):.2f}s`",
            f"**Original duration:** `{report_dict.get('original_duration', 0.0):.2f}s`",
            f"**Peak:** `{report_dict.get('peak', 0.0):.3f}`",
            f"**RMS:** `{report_dict.get('rms', 0.0):.3f}`",
            f"**Processing:** {'; '.join(operations) if operations else 'none'}",
            f"**Warnings:** {'; '.join(warnings) if warnings else 'None'}",
        ]
    )


def build_demo(runtime: AppRuntime) -> gr.Blocks:
    theme = gr.themes.Soft(font=["IBM Plex Sans", "Avenir Next", "sans-serif"])
    css = """
    .gradio-container {max-width: 1280px !important; font-size: 16px !important;}
    .hero {background: linear-gradient(135deg, #eef6ff, #f5f1e8); border-radius: 20px; padding: 20px;}
    """

    def gen_config(num_step, guidance_scale, denoise, preprocess_prompt, postprocess_output):
        return OmniVoiceGenerationConfig(
            num_step=int(num_step or 32),
            guidance_scale=float(guidance_scale or 2.0),
            denoise=bool(denoise),
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
        )

    def prepare_reference(ref_audio, ref_text, language, preprocess_prompt, auto_transcribe):
        if not ref_audio:
            return None, None, "Upload a reference clip to inspect or save it."
        approved_text = ref_text.strip() if ref_text else None
        if not approved_text and not auto_transcribe:
            return None, None, "Provide a transcript or enable auto transcription."

        prepared = runtime.model.prepare_voice_clone_reference(  # type: ignore[union-attr]
            ref_audio,
            ref_text=approved_text,
            language=_normalize_language(language),
            preprocess_prompt=bool(preprocess_prompt),
        )
        report_dict = prepared["report"].to_dict()
        state = {
            "waveform": prepared["waveform"],
            "ref_text": prepared["ref_text"],
            "language": _normalize_language(language),
            "report": report_dict,
        }
        return state, _wave_to_numpy(prepared["waveform"], runtime.model.sampling_rate), _format_reference_report(report_dict)  # type: ignore[union-attr]

    def clone_once(text, language, ref_audio, ref_text, instruct, num_step, guidance_scale, denoise, speed, duration, preprocess_prompt, postprocess_output):
        if not text or not text.strip():
            return None, "Enter the text you want to synthesize."
        if not ref_audio:
            return None, "Upload a reference audio clip first."

        prompt = runtime.model.create_voice_clone_prompt(  # type: ignore[union-attr]
            ref_audio,
            ref_text=ref_text or None,
            language=_normalize_language(language),
            preprocess_prompt=bool(preprocess_prompt),
        )
        audio = runtime.model.generate(  # type: ignore[union-attr]
            text=text.strip(),
            language=_normalize_language(language),
            voice_clone_prompt=prompt,
            instruct=instruct or None,
            speed=float(speed or 1.0),
            duration=float(duration) if duration else None,
            generation_config=gen_config(num_step, guidance_scale, denoise, preprocess_prompt, postprocess_output),
        )[0]
        return _wave_to_numpy(audio, runtime.model.sampling_rate), "Generated from the current reference clip."  # type: ignore[union-attr]

    def save_voice_profile(prepared_state, voice_name, notes):
        if not voice_name or not voice_name.strip():
            return gr.update(), gr.update(), None, "Enter a name before saving the voice."
        if not prepared_state:
            return gr.update(), gr.update(), None, "Analyze the reference clip first so the cleaned voice prompt can be saved."

        prompt = runtime.model.create_voice_clone_prompt(  # type: ignore[union-attr]
            (prepared_state["waveform"], runtime.model.sampling_rate),  # type: ignore[union-attr]
            ref_text=prepared_state["ref_text"],
            language=prepared_state.get("language"),
            preprocess_prompt=False,
        )
        profile = runtime.voice_library.create_profile(
            name=voice_name.strip(),
            cleaned_audio=prepared_state["waveform"],
            sample_rate=runtime.model.sampling_rate,  # type: ignore[union-attr]
            prompt=prompt,
            transcript=prepared_state["ref_text"],
            language=prepared_state.get("language"),
            notes=notes or "",
            metadata={"reference_report": prepared_state["report"]},
        )
        choices = _voice_choices(runtime.voice_library)
        return (
            gr.update(choices=choices, value=profile.id),
            _format_profile(profile),
            _wave_to_numpy(prepared_state["waveform"], runtime.model.sampling_rate),  # type: ignore[union-attr]
            f"Saved voice profile `{profile.display_name}`.",
        )

    def refresh_profiles(selected_id=None):
        choices = _voice_choices(runtime.voice_library)
        if not choices:
            return gr.update(choices=[], value=None), "No saved voices yet.", None
        current = selected_id if any(value == selected_id for _, value in choices) else choices[0][1]
        profile = runtime.voice_library.find_profile(current)
        ref_audio = runtime.voice_library.load_reference_audio(profile.id)
        return gr.update(choices=choices, value=current), _format_profile(profile), _wave_to_numpy(ref_audio[0], ref_audio[1])

    def load_profile_details(profile_id):
        if not profile_id:
            return "Select a saved voice profile.", None
        profile = runtime.voice_library.find_profile(profile_id)
        waveform, sample_rate = runtime.voice_library.load_reference_audio(profile.id)
        return _format_profile(profile), _wave_to_numpy(waveform, sample_rate)

    def generate_from_profile(profile_id, text, language, num_step, guidance_scale, denoise, speed, duration, postprocess_output):
        if not profile_id:
            return None, "Choose a saved voice profile first."
        if not text or not text.strip():
            return None, "Enter text to synthesize."
        prompt = runtime.voice_library.load_prompt(profile_id)
        audio = runtime.model.generate(  # type: ignore[union-attr]
            text=text.strip(),
            language=_normalize_language(language),
            voice_clone_prompt=prompt,
            speed=float(speed or 1.0),
            duration=float(duration) if duration else None,
            generation_config=OmniVoiceGenerationConfig(
                num_step=int(num_step or 32),
                guidance_scale=float(guidance_scale or 2.0),
                denoise=bool(denoise),
                postprocess_output=bool(postprocess_output),
            ),
        )[0]
        return _wave_to_numpy(audio, runtime.model.sampling_rate), "Generated from the saved voice profile."  # type: ignore[union-attr]

    def rename_profile(profile_id, new_name):
        if not profile_id or not new_name.strip():
            return gr.update(), gr.update(), "Choose a profile and enter a new name."
        profile = runtime.voice_library.rename_profile(profile_id, new_name.strip())
        choices = _voice_choices(runtime.voice_library)
        return gr.update(choices=choices, value=profile.id), _format_profile(profile), "Voice renamed."

    def delete_profile(profile_id):
        if not profile_id:
            return gr.update(), "Choose a profile to delete.", None
        runtime.voice_library.delete_profile(profile_id)
        return refresh_profiles()

    def export_profile(profile_id):
        if not profile_id:
            return None, "Choose a profile to export."
        path = runtime.voice_library.export_profile(
            profile_id, runtime.voice_library.root_dir / f"{profile_id}.zip"
        )
        return str(path), "Exported voice profile."

    def import_profile(archive):
        if not archive:
            return gr.update(), "Choose a `.zip` voice profile export to import.", None
        profile = runtime.voice_library.import_profile(archive)
        dropdown, details, ref_audio = refresh_profiles(profile.id)
        return dropdown, f"Imported `{profile.display_name}`.\n\n{details}", ref_audio

    def reload_backend(device_override, asr_enabled):
        runtime.reload(device_override, load_asr=bool(asr_enabled))
        dropdown, details, ref_audio = refresh_profiles()
        return runtime.summarize_runtime(), dropdown, details, ref_audio

    def benchmark_backend():
        start = time.perf_counter()
        audio = runtime.model.generate(  # type: ignore[union-attr]
            text="This is a short backend benchmark for OmniVoice.",
            generation_config=OmniVoiceGenerationConfig(num_step=8, guidance_scale=1.5),
        )[0]
        elapsed = time.perf_counter() - start
        return _wave_to_numpy(audio, runtime.model.sampling_rate), f"Completed benchmark in {elapsed:.2f}s."  # type: ignore[union-attr]

    def _build_instruct(groups):
        selected = [g for g in groups if g and g != "Auto"]
        if not selected:
            return None
        parts = []
        for value in selected:
            if " / " in value:
                en, zh = value.split(" / ", 1)
                parts.append(zh.strip() if "Dialect" in en else en.strip())
            else:
                parts.append(value)
        return ", ".join(parts)

    def generate_designed_voice(text, language, num_step, guidance_scale, denoise, speed, duration, postprocess_output, *groups):
        if not text or not text.strip():
            return None, "Enter text to synthesize."
        audio = runtime.model.generate(  # type: ignore[union-attr]
            text=text.strip(),
            language=_normalize_language(language),
            instruct=_build_instruct(groups),
            speed=float(speed or 1.0),
            duration=float(duration) if duration else None,
            generation_config=OmniVoiceGenerationConfig(
                num_step=int(num_step or 32),
                guidance_scale=float(guidance_scale or 2.0),
                denoise=bool(denoise),
                postprocess_output=bool(postprocess_output),
            ),
        )[0]
        return _wave_to_numpy(audio, runtime.model.sampling_rate), "Generated with voice design."  # type: ignore[union-attr]

    with gr.Blocks(theme=theme, css=css, title="OmniVoice Mac Demo") as demo:
        prepared_state = gr.State(None)

        gr.Markdown(
            """
<div class="hero">

# OmniVoice for macOS

Clone a voice from a short clean clip, save it locally as a reusable profile, and come back later without re-uploading the reference audio.

</div>
"""
        )

        runtime_markdown = gr.Markdown(runtime.summarize_runtime())

        with gr.Tabs():
            with gr.TabItem("Voice Clone"):
                with gr.Row():
                    with gr.Column(scale=1):
                        clone_text = gr.Textbox(label="Text to Synthesize", lines=4)
                        ref_audio = gr.Audio(label="Reference Audio", type="filepath")
                        gr.Markdown(
                            "Use a clean **3-10 second** single-speaker clip. Avoid echo, music, clipping, and long pauses."
                        )
                        ref_text = gr.Textbox(
                            label="Reference Transcript",
                            lines=3,
                            placeholder="Manual transcript is best. Leave blank to auto-transcribe if ASR is enabled.",
                        )
                        clone_lang = _lang_dropdown()
                        auto_transcribe = gr.Checkbox(
                            label="Auto transcribe when transcript is blank",
                            value=runtime.load_asr,
                        )
                        voice_name = gr.Textbox(label="Voice Profile Name", placeholder="Narrator A")
                        voice_notes = gr.Textbox(label="Notes (optional)", lines=2)
                        with gr.Accordion("Advanced Generation Controls", open=False):
                            clone_instruct = gr.Textbox(label="Optional Voice Instruction", lines=2)
                            clone_num_step = gr.Slider(4, 64, value=32, step=1, label="Inference Steps")
                            clone_guidance = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="Guidance Scale")
                            clone_denoise = gr.Checkbox(label="Denoise", value=True)
                            clone_speed = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Speed")
                            clone_duration = gr.Number(value=None, label="Fixed Duration (seconds)")
                            clone_preprocess = gr.Checkbox(label="Preprocess Reference", value=True)
                            clone_postprocess = gr.Checkbox(label="Postprocess Output", value=True)
                        with gr.Row():
                            analyze_btn = gr.Button("Analyze Reference")
                            save_btn = gr.Button("Save Voice Profile", variant="secondary")
                            clone_btn = gr.Button("Generate Once", variant="primary")
                    with gr.Column(scale=1):
                        cleaned_ref_audio = gr.Audio(label="Cleaned Reference Preview", type="numpy")
                        preprocess_report = gr.Markdown("Analyze a clip to see preprocessing details and warnings.")
                        clone_output = gr.Audio(label="Generated Audio", type="numpy")
                        clone_status = gr.Textbox(label="Status", lines=3)

            with gr.TabItem("Saved Voices"):
                with gr.Row():
                    with gr.Column(scale=1):
                        voice_dropdown = gr.Dropdown(
                            label="Saved Voice Profiles",
                            choices=_voice_choices(runtime.voice_library),
                            value=_voice_choices(runtime.voice_library)[0][1] if _voice_choices(runtime.voice_library) else None,
                        )
                        refresh_btn = gr.Button("Refresh Voices")
                        saved_voice_text = gr.Textbox(label="Text to Synthesize", lines=4)
                        saved_voice_lang = _lang_dropdown()
                        rename_value = gr.Textbox(label="Rename Selected Voice")
                        import_file = gr.File(label="Import Voice Profile (.zip)", type="filepath")
                        with gr.Accordion("Advanced Generation Controls", open=False):
                            saved_num_step = gr.Slider(4, 64, value=32, step=1, label="Inference Steps")
                            saved_guidance = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="Guidance Scale")
                            saved_denoise = gr.Checkbox(label="Denoise", value=True)
                            saved_speed = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Speed")
                            saved_duration = gr.Number(value=None, label="Fixed Duration (seconds)")
                            saved_postprocess = gr.Checkbox(label="Postprocess Output", value=True)
                        with gr.Row():
                            generate_saved_btn = gr.Button("Generate From Saved Voice", variant="primary")
                            rename_btn = gr.Button("Rename")
                            delete_btn = gr.Button("Delete")
                            export_btn = gr.Button("Export")
                            import_btn = gr.Button("Import")
                    with gr.Column(scale=1):
                        profile_details = gr.Markdown("No saved voices yet.")
                        profile_reference_audio = gr.Audio(label="Saved Reference Preview", type="numpy")
                        saved_voice_output = gr.Audio(label="Generated Audio", type="numpy")
                        saved_status = gr.Textbox(label="Status", lines=3)
                        export_file = gr.File(label="Exported Voice Archive")


            with gr.TabItem("Voice Design"):
                with gr.Row():
                    with gr.Column(scale=1):
                        design_text = gr.Textbox(label="Text to Synthesize", lines=4)
                        design_lang = _lang_dropdown()
                        design_groups = [
                            gr.Dropdown(
                                label=category,
                                choices=["Auto"] + choices,
                                value="Auto",
                                info=_ATTR_INFO.get(category),
                            )
                            for category, choices in _CATEGORIES.items()
                        ]
                        with gr.Accordion("Advanced Generation Controls", open=False):
                            design_num_step = gr.Slider(4, 64, value=32, step=1, label="Inference Steps")
                            design_guidance = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="Guidance Scale")
                            design_denoise = gr.Checkbox(label="Denoise", value=True)
                            design_speed = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Speed")
                            design_duration = gr.Number(value=None, label="Fixed Duration (seconds)")
                            design_postprocess = gr.Checkbox(label="Postprocess Output", value=True)
                        design_btn = gr.Button("Generate Designed Voice", variant="primary")
                    with gr.Column(scale=1):
                        design_audio = gr.Audio(label="Output Audio", type="numpy")
                        design_status = gr.Textbox(label="Status", lines=2)
                design_btn.click(
                    generate_designed_voice,
                    inputs=[
                        design_text,
                        design_lang,
                        design_num_step,
                        design_guidance,
                        design_denoise,
                        design_speed,
                        design_duration,
                        design_postprocess,
                    ]
                    + design_groups,
                    outputs=[design_audio, design_status],
                )

            with gr.TabItem("Diagnostics / Settings"):
                backend_override = gr.Dropdown(
                    label="Backend Override",
                    choices=["auto"] + available_backends(),
                    value="auto",
                )
                asr_toggle = gr.Checkbox(label="Enable ASR auto transcription", value=runtime.load_asr)
                diagnostics_btn = gr.Button("Reload Backend / Apply Settings")
                benchmark_btn = gr.Button("Quick Benchmark")
                benchmark_audio = gr.Audio(label="Benchmark Output", type="numpy")
                benchmark_status = gr.Textbox(label="Benchmark Status", lines=2)
                diagnostics_raw = gr.Code(
                    value=json.dumps(runtime.runtime_info, indent=2),
                    label="Raw Runtime Metadata",
                    language="json",
                )

        analyze_btn.click(
            prepare_reference,
            inputs=[ref_audio, ref_text, clone_lang, clone_preprocess, auto_transcribe],
            outputs=[prepared_state, cleaned_ref_audio, preprocess_report],
        ).then(
            lambda state: state["ref_text"] if state else gr.update(),
            inputs=[prepared_state],
            outputs=[ref_text],
        )
        save_btn.click(
            save_voice_profile,
            inputs=[prepared_state, voice_name, voice_notes],
            outputs=[voice_dropdown, profile_details, profile_reference_audio, clone_status],
        )
        clone_btn.click(
            clone_once,
            inputs=[
                clone_text,
                clone_lang,
                ref_audio,
                ref_text,
                clone_instruct,
                clone_num_step,
                clone_guidance,
                clone_denoise,
                clone_speed,
                clone_duration,
                clone_preprocess,
                clone_postprocess,
            ],
            outputs=[clone_output, clone_status],
        )
        voice_dropdown.change(load_profile_details, inputs=[voice_dropdown], outputs=[profile_details, profile_reference_audio])
        refresh_btn.click(refresh_profiles, inputs=[voice_dropdown], outputs=[voice_dropdown, profile_details, profile_reference_audio])
        generate_saved_btn.click(
            generate_from_profile,
            inputs=[
                voice_dropdown,
                saved_voice_text,
                saved_voice_lang,
                saved_num_step,
                saved_guidance,
                saved_denoise,
                saved_speed,
                saved_duration,
                saved_postprocess,
            ],
            outputs=[saved_voice_output, saved_status],
        )
        rename_btn.click(
            rename_profile,
            inputs=[voice_dropdown, rename_value],
            outputs=[voice_dropdown, profile_details, saved_status],
        )
        delete_btn.click(delete_profile, inputs=[voice_dropdown], outputs=[voice_dropdown, profile_details, profile_reference_audio])
        export_btn.click(export_profile, inputs=[voice_dropdown], outputs=[export_file, saved_status])
        import_btn.click(import_profile, inputs=[import_file], outputs=[voice_dropdown, profile_details, profile_reference_audio])
        diagnostics_btn.click(
            reload_backend,
            inputs=[backend_override, asr_toggle],
            outputs=[runtime_markdown, voice_dropdown, profile_details, profile_reference_audio],
        ).then(
            lambda: json.dumps(runtime.runtime_info, indent=2),
            outputs=[diagnostics_raw],
        )
        benchmark_btn.click(benchmark_backend, outputs=[benchmark_audio, benchmark_status])
        demo.load(refresh_profiles, outputs=[voice_dropdown, profile_details, profile_reference_audio])

    return demo


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    runtime = AppRuntime(
        checkpoint=args.model,
        device=args.device,
        load_asr=not args.no_asr,
        voices_dir=args.voices_dir,
    )
    logger.info("Runtime info: %s", json.dumps(runtime.runtime_info, indent=2))

    demo = build_demo(runtime)
    demo.queue().launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        root_path=args.root_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
