"""OneDrive sync: find new or changed files, process each in a temp folder, delete the temp copy.

Changes are tracked by item id. Graph delta responses omit ``parentReference.path`` and
their download URLs expire quickly, so each changed file is re-read with a normal GET
(fresh path, hash and download URL) just before it is processed.

Delta on a sub-folder is supported on personal OneDrive but not always on OneDrive for
Business. When delta fails the folder is scanned in full instead; unchanged files are
still skipped by hash without being downloaded, but deletions cannot be detected that way.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable
from urllib.parse import unquote

from ..config import Config, FolderConfig
from ..extract import is_supported
from ..index.embedder import Embedder
from ..index.store import Store
from ..pipeline import ProcessResult, process_file
from ..privacy.classifier import PathRule
from .graph import GraphClient, GraphError

log = logging.getLogger("texti")

ProcessFn = Callable[..., ProcessResult]


@dataclass
class SyncStats:
    processed: int = 0
    unchanged: int = 0
    skipped: int = 0  # unsupported type, folder, or too large
    deleted: int = 0
    failed: int = 0
    full_scan: bool = False
    errors: list[str] = field(default_factory=list)  # exception type names only, never paths or text


def _drive_path(parent_reference: dict | None) -> str:
    """Folder path from a parentReference, relative to the drive root, e.g. 'College/Maths'."""
    raw = (parent_reference or {}).get("path") or ""
    if "root:" not in raw:
        return ""
    return unquote(raw.split("root:", 1)[1]).strip("/")


def item_path(item: dict) -> str:
    return "/".join(p for p in (_drive_path(item.get("parentReference")), item.get("name", "")) if p)


def content_hash(item: dict) -> str | None:
    hashes = (item.get("file") or {}).get("hashes") or {}
    for algo in ("sha256Hash", "sha1Hash", "quickXorHash"):
        if hashes.get(algo):
            return f"{algo}:{hashes[algo]}"
    tag = item.get("cTag") or item.get("eTag")
    return f"tag:{tag}" if tag else None


def list_folders(client: GraphClient, parent_id: str = "root") -> list[dict]:
    """Sub-folders of a folder, for the folder picker."""
    path = "/me/drive/root/children" if parent_id == "root" else f"/me/drive/items/{parent_id}/children"
    out: list[dict] = []
    for page in client.pages(f"{path}?$select=id,name,folder&$top=200"):
        out.extend(i for i in page.get("value", []) if "folder" in i)
    return sorted(out, key=lambda i: i.get("name", "").lower())


def _iter_delta(client: GraphClient, folder_id: str, delta_link: str | None):
    """Yield change stubs; the generator's return value is the new delta link."""
    url = delta_link or f"/me/drive/items/{folder_id}/delta?$select=id,name,file,folder,deleted,size"
    new_link = None
    for page in client.pages(url):
        for item in page.get("value", []):
            yield item
        new_link = page.get("@odata.deltaLink", new_link)
    return new_link


def _iter_full_scan(client: GraphClient, folder_id: str):
    """Recursive listing of every file under a folder."""
    stack = [folder_id]
    while stack:
        current = stack.pop()
        for page in client.pages(f"/me/drive/items/{current}/children?$select=id,name,file,folder,size&$top=200"):
            for item in page.get("value", []):
                if "folder" in item:
                    stack.append(item["id"])
                else:
                    yield item


def sync_folder(
    client: GraphClient,
    store: Store,
    embedder: Embedder,
    cfg: Config,
    folder: FolderConfig,
    process: ProcessFn = process_file,
) -> SyncStats:
    stats = SyncStats()
    root_item = client.get_json(f"/me/drive/items/{folder.id}?$select=id,name,parentReference")
    root_path = item_path(root_item)  # full OneDrive path of the picked folder
    # The folder passed in is authoritative: its own tier rule is appended last, so it wins any
    # tie with an older copy of the same folder in the saved config.
    rules = cfg.rules() + ([PathRule(folder.name, folder.tier)] if folder.tier else [])

    saved_link = store.get_delta_link(folder.id)
    changes = []
    new_link: str | None = None
    try:
        gen = _iter_delta(client, folder.id, saved_link)
        while True:
            try:
                changes.append(next(gen))
            except StopIteration as stop:
                new_link = stop.value
                break
    except GraphError as e:
        log.info("delta unavailable for folder (HTTP %s); scanning in full", e.status)
        stats.full_scan = True
        changes = list(_iter_full_scan(client, folder.id))
        new_link = None

    for stub in changes:
        item_id = stub.get("id", "")
        if stub.get("deleted"):
            store.delete_file(item_id)
            stats.deleted += 1
            continue
        if "folder" in stub or "file" not in stub or not is_supported(stub.get("name", "")):
            stats.skipped += 1
            continue
        if stub.get("size", 0) > cfg.max_file_mb * 1024 * 1024:
            stats.skipped += 1
            continue
        try:
            _sync_one(client, store, embedder, cfg, folder, rules, root_path, item_id, stats, process)
        except Exception as e:  # noqa: BLE001 - one bad file must not stop the sync
            stats.failed += 1
            stats.errors.append(type(e).__name__)  # never the message: it could contain a path
            log.warning("file_id=%s failed: %s", item_id, type(e).__name__)

    # Keep the old delta link if anything failed, so the failed files are retried next time.
    if stats.failed == 0 and not stats.full_scan:
        store.set_delta_link(folder.id, new_link)
    return stats


def _sync_one(client, store, embedder, cfg, folder, rules, root_path, item_id, stats, process) -> None:
    meta = client.get_json(
        f"/me/drive/items/{item_id}?$select=id,name,size,file,cTag,eTag,parentReference,@microsoft.graph.downloadUrl"
    )
    remote_hash = content_hash(meta)
    if remote_hash and store.file_hash(item_id) == remote_hash:
        stats.unchanged += 1
        return

    full = item_path(meta)
    try:
        inner = PurePosixPath(full).relative_to(PurePosixPath(root_path)).as_posix() if root_path else full
    except ValueError:  # path casing or encoding differs from the picked folder's: fall back to the name
        inner = meta["name"]
    rel_path = f"{folder.name}/{inner}"

    # The temp folder is private to this user (mode 0700) and is removed even on failure.
    with tempfile.TemporaryDirectory(prefix="m2t-") as tmp:
        local = Path(tmp) / f"download{PurePosixPath(meta['name']).suffix.lower()}"
        client.download(item_id, meta.get("@microsoft.graph.downloadUrl"), local)
        result = process(
            store, embedder, cfg, path=local, file_id=item_id, rel_path=rel_path,
            content_hash=remote_hash or _sha256(local), language=folder.language, rules=rules,
        )
    if result.status == "unchanged":
        stats.unchanged += 1
    else:
        stats.processed += 1


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return f"sha256:{h.hexdigest()}"
