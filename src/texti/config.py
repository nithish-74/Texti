"""User configuration, stored as JSON in the data directory (default ~/.texti)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .index.embedder import DEFAULT_MODEL
from .privacy.classifier import PathRule

_migration_warned = False


def data_dir() -> Path:
    global _migration_warned
    if "TEXTI_HOME" in os.environ:
        return Path(os.environ["TEXTI_HOME"])
    new_dir = Path.home() / ".texti"
    old_dir = Path.home() / ".media2text"
    if old_dir.exists() and not new_dir.exists() and not _migration_warned:
        print("Notice: ~/.media2text exists and ~/.texti does not. Please move ~/.media2text to ~/.texti.")
        _migration_warned = True
    return new_dir


@dataclass
class FolderConfig:
    id: str
    name: str
    language: str = "eng"  # Tesseract language code(s), e.g. "eng", "eng+tel", "hin"
    tier: int | None = None  # None: no folder rule, files start at default_tier


def _folder_from_raw(raw) -> "FolderConfig | None":
    """Build a FolderConfig from saved JSON, tolerating older or foreign settings files
    (for example a `folder_id` key instead of `id`). Entries that cannot be used are skipped."""
    if not isinstance(raw, dict):
        return None
    folder_id = raw.get("id") or raw.get("folder_id")
    name = raw.get("name") or raw.get("folder_name")
    if not folder_id or not name:
        return None
    tier = raw.get("tier")
    return FolderConfig(
        id=str(folder_id),
        name=str(name),
        language=str(raw.get("language") or "eng"),
        tier=tier if tier in (1, 2, 3) else None,
    )


@dataclass
class Config:
    folders: list[FolderConfig] = field(default_factory=list)
    path_rules: list[dict] = field(default_factory=list)  # [{"prefix": "Finance", "tier": 3}]
    default_tier: int = 2
    budget: int = 800
    min_confidence: float = 0.6
    min_score: float = 0.1
    max_files: int = 5
    embedding_model: str = DEFAULT_MODEL
    image_tokens: int = 765  # estimated cost of one image sent to a vision model
    use_tiktoken: bool = False  # tiktoken downloads its vocabulary on first use: opt in
    max_file_mb: int = 50
    target_tokens: int = 400
    max_tokens: int = 500
    overlap_tokens: int = 50
    hybrid_weight: float = 0.3


    def rules(self) -> list[PathRule]:
        rules = [PathRule(r["prefix"], int(r["tier"])) for r in self.path_rules]
        rules += [PathRule(f.name, f.tier) for f in self.folders if f.tier]
        return rules

    def folder_by_id(self, folder_id: str) -> FolderConfig | None:
        return next((f for f in self.folders if f.id == folder_id), None)

    # -- persistence ---------------------------------------------------------------

    @classmethod
    def load(cls, directory: Path | None = None) -> "Config":
        path = (directory or data_dir()) / "config.json"
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("config is not an object")
        except (ValueError, OSError):
            # Unreadable settings: keep the file as a backup and start fresh.
            try:
                path.replace(path.with_name("config.json.bad"))
            except OSError:
                pass
            return cls()
        folders = [f for f in (_folder_from_raw(x) for x in raw.pop("folders", []) or []) if f]
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        return cls(folders=folders, **known)

    def save(self, directory: Path | None = None) -> Path:
        d = directory or data_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / "config.json"
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        return path
