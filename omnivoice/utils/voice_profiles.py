"""Reusable local voice profile storage."""

from __future__ import annotations

import json
import shutil
import tempfile
import uuid
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Optional

import torch

from omnivoice.models.omnivoice import VoiceClonePrompt
from omnivoice.utils.audio import load_audio_file_any, save_audio_file_any
from omnivoice.utils.app_paths import get_voice_library_dir

SCHEMA_VERSION = 1
PROFILE_JSON = "profile.json"
CONDITIONING_FILE = "conditioning.pt"
REFERENCE_AUDIO_FILE = "reference.wav"

try:
    APP_VERSION = version("omnivoice")
except PackageNotFoundError:
    APP_VERSION = "0.0.0"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify_name(name: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in name.strip())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned or "voice"


@dataclass
class VoiceProfile:
    id: str
    display_name: str
    created_at: str
    updated_at: str
    transcript: str
    language: Optional[str]
    duration_seconds: float
    notes: str
    tags: list[str]
    sample_rate: int
    schema_version: int
    app_version: str
    audio_filename: str = REFERENCE_AUDIO_FILE
    conditioning_filename: str = CONDITIONING_FILE
    metadata: Optional[dict[str, Any]] = None


class VoiceLibrary:
    def __init__(self, root_dir: Optional[str | Path] = None):
        self.root_dir = Path(root_dir) if root_dir else get_voice_library_dir()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _profile_dir(self, profile_id: str) -> Path:
        return self.root_dir / profile_id

    def _profile_json_path(self, profile_id: str) -> Path:
        return self._profile_dir(profile_id) / PROFILE_JSON

    def _conditioning_path(self, profile_id: str) -> Path:
        return self._profile_dir(profile_id) / CONDITIONING_FILE

    def _audio_path(self, profile_id: str) -> Path:
        return self._profile_dir(profile_id) / REFERENCE_AUDIO_FILE

    def create_profile(
        self,
        *,
        name: str,
        cleaned_audio: torch.Tensor,
        sample_rate: int,
        prompt: VoiceClonePrompt,
        transcript: str,
        language: Optional[str] = None,
        notes: str = "",
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> VoiceProfile:
        profile_id = f"{slugify_name(name)}-{uuid.uuid4().hex[:8]}"
        profile_dir = self._profile_dir(profile_id)
        profile_dir.mkdir(parents=True, exist_ok=False)

        save_audio_file_any(str(self._audio_path(profile_id)), cleaned_audio, sample_rate)
        torch.save(
            {
                "ref_audio_tokens": prompt.ref_audio_tokens.cpu(),
                "ref_rms": float(prompt.ref_rms),
                "ref_text": prompt.ref_text,
            },
            self._conditioning_path(profile_id),
        )

        duration = float(cleaned_audio.shape[-1] / max(sample_rate, 1))
        profile = VoiceProfile(
            id=profile_id,
            display_name=name.strip(),
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
            transcript=transcript,
            language=language,
            duration_seconds=duration,
            notes=notes.strip(),
            tags=list(tags or []),
            sample_rate=sample_rate,
            schema_version=SCHEMA_VERSION,
            app_version=APP_VERSION,
            metadata=metadata or {},
        )
        self._write_profile(profile)
        return profile

    def _write_profile(self, profile: VoiceProfile) -> None:
        self._profile_json_path(profile.id).write_text(
            json.dumps(asdict(profile), indent=2, sort_keys=True)
        )

    def load_profile(self, profile_id: str) -> VoiceProfile:
        return VoiceProfile(**json.loads(self._profile_json_path(profile_id).read_text()))

    def list_profiles(self) -> list[VoiceProfile]:
        profiles = []
        for path in sorted(self.root_dir.glob(f"*/{PROFILE_JSON}")):
            profiles.append(VoiceProfile(**json.loads(path.read_text())))
        profiles.sort(key=lambda profile: profile.updated_at, reverse=True)
        return profiles

    def find_profile(self, name_or_id: str) -> VoiceProfile:
        name_or_id = name_or_id.strip()
        for profile in self.list_profiles():
            if profile.id == name_or_id or profile.display_name == name_or_id:
                return profile
        raise FileNotFoundError(f"Voice profile not found: {name_or_id}")

    def load_prompt(self, name_or_id: str) -> VoiceClonePrompt:
        profile = self.find_profile(name_or_id)
        data = torch.load(self._conditioning_path(profile.id), map_location="cpu")
        return VoiceClonePrompt(
            ref_audio_tokens=data["ref_audio_tokens"],
            ref_text=data["ref_text"],
            ref_rms=float(data["ref_rms"]),
        )

    def load_reference_audio(self, name_or_id: str) -> tuple[torch.Tensor, int]:
        profile = self.find_profile(name_or_id)
        return load_audio_file_any(str(self._audio_path(profile.id)))

    def rename_profile(self, name_or_id: str, new_name: str) -> VoiceProfile:
        profile = self.find_profile(name_or_id)
        profile.display_name = new_name.strip()
        profile.updated_at = utc_now_iso()
        self._write_profile(profile)
        return profile

    def delete_profile(self, name_or_id: str) -> VoiceProfile:
        profile = self.find_profile(name_or_id)
        shutil.rmtree(self._profile_dir(profile.id))
        return profile

    def export_profile(self, name_or_id: str, output_path: str | Path) -> Path:
        profile = self.find_profile(name_or_id)
        profile_dir = self._profile_dir(profile.id)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for child in profile_dir.iterdir():
                archive.write(child, arcname=f"{profile.id}/{child.name}")
        return output_path

    def import_profile(self, archive_path: str | Path, *, replace: bool = False) -> VoiceProfile:
        archive_path = Path(archive_path)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive.extractall(tmp_path)

            profile_jsons = list(tmp_path.glob(f"*/{PROFILE_JSON}"))
            if len(profile_jsons) != 1:
                raise ValueError("Archive must contain exactly one voice profile.")

            extracted_dir = profile_jsons[0].parent
            profile = VoiceProfile(**json.loads(profile_jsons[0].read_text()))
            target_dir = self._profile_dir(profile.id)
            if target_dir.exists():
                if not replace:
                    raise FileExistsError(
                        f"Voice profile already exists: {profile.display_name} ({profile.id})"
                    )
                shutil.rmtree(target_dir)
            shutil.copytree(extracted_dir, target_dir)
        return self.load_profile(profile.id)
