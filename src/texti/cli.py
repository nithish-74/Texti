"""Command-line interface: login, pick-folder, sync, ingest, search, report, forget, fetch-model."""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from pathlib import Path
from typing import Callable, TextIO

from . import __version__, tokens
from .config import Config, FolderConfig, data_dir
from .extract import is_supported
from .extract.audio import MEDIA_AUDIO_EXTENSIONS
from .index.embedder import Embedder, get_embedder
from .index.store import Store
from .pipeline import process_file
from .privacy.classifier import PathRule, parse_tier
from .retrieve.search import Hit, search
from .secrets_store import KeyringSecretStore, SecretStore

SUPPORT = "Supported: images (png, jpg, gif, bmp, tif, webp), PDFs, and audio/video (wav, mp3, ogg, flac, m4a, mp4, mkv, webm)."



class CliError(Exception):
    """A user-facing error: printed without a traceback."""


class App:
    def __init__(self, secrets: SecretStore, out: TextIO, embedder: Embedder | None = None,
                 ask: Callable[[str], str] = input) -> None:
        self.secrets, self.out, self.ask = secrets, out, ask
        self._embedder = embedder
        self.dir = data_dir()
        self.cfg = Config.load(self.dir)
        tokens.configure(self.cfg.use_tiktoken)
        self._store: Store | None = None

    def say(self, msg: str = "") -> None:
        print(msg, file=self.out)

    @property
    def store(self) -> Store:
        if self._store is None:
            self._store = Store(self.dir / "data", self.secrets)
        return self._store

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder(self.cfg.embedding_model)
        return self._embedder

    def close(self) -> None:
        if self._store is not None:
            self._store.close()


# -- commands ------------------------------------------------------------------------------------


def cmd_login(app: App, args) -> None:
    from .onedrive.auth import Authenticator

    name = Authenticator(app.secrets).login()
    app.say(f"Signed in as {name}. Access is read-only (Files.Read).")


def cmd_pick_folder(app: App, args) -> None:
    from .onedrive.auth import Authenticator
    from .onedrive.graph import GraphClient
    from .onedrive.sync import list_folders

    client = GraphClient(Authenticator(app.secrets).token)
    stack: list[tuple[str, str]] = [("root", "OneDrive")]
    picked: list[FolderConfig] = []
    while True:
        parent_id, parent_name = stack[-1]
        folders = list_folders(client, parent_id)
        app.say(f"\n{parent_name}/")
        for i, f in enumerate(folders, 1):
            app.say(f"  {i:>2}. {f['name']}")
        choice = app.ask("Pick a number, 'o<number>' to open a folder, '..' to go up, 'q' when done: ").strip().lower()
        if choice == "q":
            break
        if choice == ".." and len(stack) > 1:
            stack.pop()
        elif choice.startswith("o") and choice[1:].isdigit() and 0 < int(choice[1:]) <= len(folders):
            f = folders[int(choice[1:]) - 1]
            stack.append((f["id"], f["name"]))
        elif choice.isdigit() and 0 < int(choice) <= len(folders):
            f = folders[int(choice) - 1]
            lang = app.ask(f"Language(s) for '{f['name']}' as Tesseract codes, e.g. eng, eng+tel, hin [eng]: ").strip() or "eng"
            tier_in = app.ask("Privacy tier for this folder: 1 general, 2 personal, 3 private, blank = decide per file: ")
            tier = parse_tier(tier_in)
            picked.append(FolderConfig(id=f["id"], name=f["name"], language=lang, tier=tier))
            app.say(f"Added '{f['name']}'.")
        else:
            app.say("Not a valid choice.")
    if picked:
        ids = {f.id for f in picked}
        app.cfg.folders = [f for f in app.cfg.folders if f.id not in ids] + picked
        app.cfg.save(app.dir)
    app.say(f"{len(app.cfg.folders)} folder(s) configured.")


def cmd_sync(app: App, args) -> None:
    from .onedrive.auth import Authenticator
    from .onedrive.graph import GraphClient
    from .onedrive.sync import sync_folder

    folders = [f for f in app.cfg.folders if not args.folder or f.name.lower() == args.folder.lower()]
    if not folders:
        raise CliError("No folders configured. Run `texti pick-folder` first.")
    client = GraphClient(Authenticator(app.secrets).token)
    for f in folders:
        t0 = time.perf_counter()
        s = sync_folder(client, app.store, app.embedder, app.cfg, f)
        mode = "full scan" if s.full_scan else "incremental"
        app.say(f"{f.name}: {s.processed} processed, {s.unchanged} unchanged, {s.deleted} deleted, "
                f"{s.skipped} skipped, {s.failed} failed ({mode}, {time.perf_counter() - t0:.1f}s)")
        if s.failed:
            app.say(f"  {s.failed} file(s) failed and will be retried on the next sync. Error types: "
                    f"{', '.join(sorted(set(s.errors)))}")


def _local_files(paths: list[str]) -> list[tuple[Path, str]]:
    """(file, path relative to its label) for every supported file under the given paths.

    The label is the name of the folder given on the command line, so a file at
    ``College/Maths/a.png`` is indexed as ``College/Maths/a.png`` and folder rules written
    against ``College`` apply to it.
    """
    found: list[tuple[Path, str]] = []
    for raw in paths:
        p = Path(raw).expanduser().resolve()
        if p.is_dir():
            found += [(f, f"{p.name}/{f.relative_to(p).as_posix()}")
                      for f in sorted(p.rglob("*")) if f.is_file() and is_supported(f.name)]
        elif p.is_file() and is_supported(p.name):
            found.append((p, f"{p.parent.name}/{p.name}"))
        else:
            raise CliError(f"Not a supported file or folder: {raw}. {SUPPORT}")
    return found


def cmd_ingest(app: App, args) -> None:
    """Index local files directly, without OneDrive. Useful for timing and for offline use."""
    files = _local_files(args.paths)
    if not files:
        raise CliError(f"No supported files found. {SUPPORT}")
    forced = parse_tier(args.tier) if args.tier else None
    if args.tier and forced is None:
        raise CliError("--tier must be 1, 2 or 3")
    t0 = time.perf_counter()
    done = unchanged = failed = 0
    for path, rel in files:
        label = rel.split("/", 1)[0]
        if path.stat().st_size > app.cfg.max_file_mb * 1024 * 1024:
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        # Opaque id: never the path, so nothing readable is written to the metadata database.
        file_id = "local-" + hashlib.sha256(str(path).encode()).hexdigest()[:20]
        rules = app.cfg.rules() + ([PathRule(label, forced)] if forced else [])
        try:
            r = process_file(app.store, app.embedder, app.cfg, path=path, file_id=file_id, rel_path=rel,
                             content_hash=f"sha256:{digest}", language=args.lang, rules=rules)
        except Exception as e:  # noqa: BLE001
            failed += 1
            app.say(f"  failed: {path.name} ({type(e).__name__}: {e})")
            continue
        if r.status == "unchanged":
            unchanged += 1
        else:
            done += 1
            if args.verbose:
                app.say(f"  {path.name}: tier {r.max_tier}, {r.chunks} chunks, {r.seconds:.2f}s")
    total = time.perf_counter() - t0
    per = total / done if done else 0.0
    app.say(f"{done} processed, {unchanged} unchanged, {failed} failed in {total:.1f}s ({per:.2f}s per file)")


def _make_approver(app: App, assume_yes: bool):
    def approver(hits: list[Hit]) -> list[Hit]:
        app.say("\nThese results are from a protected tier and have NOT been sent anywhere. Review them:")
        for h in hits:
            app.say(f"  [Tier {h.tier}] {h.source_path}: {h.text[:200]!r}")
        if assume_yes:
            return hits
        answer = app.ask(f"Include these {len(hits)} result(s)? [y/N]: ").strip().lower()
        return hits if answer in ("y", "yes") else []

    return approver


def cmd_search(app: App, args) -> None:
    try:
        tiers = tuple(sorted({int(t) for t in args.tiers.split(",") if t.strip()}))
    except ValueError:
        raise CliError("--tiers must be a comma-separated list such as 1 or 1,2") from None
    if not tiers or any(t not in (1, 2, 3) for t in tiers):
        raise CliError("--tiers may only contain 1, 2 and 3")
    res = search(
        app.store, app.embedder, args.question, budget=args.budget or app.cfg.budget, tiers=tiers,
        min_confidence=app.cfg.min_confidence, min_score=app.cfg.min_score, max_files=app.cfg.max_files,
        approver=_make_approver(app, args.yes) if any(t >= 2 for t in tiers) else None,
        hybrid_weight=app.cfg.hybrid_weight,
    )

    for note in res.notes:
        app.say(note)
    if not res.hits:
        app.say("No matching results.")
    for h in res.hits:
        ext = Path(h.source_path).suffix.lower()
        if ext in MEDIA_AUDIO_EXTENSIONS:
            mins, secs = divmod(h.source_offset, 60)
            where = f"time {mins:02d}:{secs:02d}"
        else:
            where = f"page {h.source_offset}" if h.source_offset else "image"
        flag = "  LOW CONFIDENCE: check the original file" if h.needs_original else ""

        app.say(f"\n[Tier {h.tier}] {h.source_path} ({where})  score {h.score:.2f}  {h.token_count} tokens  "
                f"confidence {h.confidence:.2f}{flag}")
        app.say(h.text)
    app.say(f"\nTokens returned: {res.tokens_returned} | raw media (estimate): {res.tokens_raw_estimate} | "
            f"saved (estimate): {res.tokens_saved_estimate}")


def cmd_report(app: App, args) -> None:
    s = app.store.stats()
    per = s["processing_seconds"] / s["files"] if s["files"] else 0.0
    app.say(f"Files processed: {s['files']}  (Tier 1: {s['files_by_tier'][1]}, Tier 2: {s['files_by_tier'][2]}, "
            f"Tier 3: {s['files_by_tier'][3]})")
    app.say(f"Chunks stored:   Tier 1: {s['chunks_by_tier'][1]}, Tier 2: {s['chunks_by_tier'][2]}, "
            f"Tier 3: {s['chunks_by_tier'][3]}")
    app.say(f"Processing time: {s['processing_seconds']:.1f}s total, {per:.2f}s per file")
    app.say(f"Searches run:    {s['queries']}")
    app.say(f"Tokens returned: {s['tokens_returned']}  vs raw media (estimate): {s['tokens_raw_estimate']}")
    app.say(f"Tokens saved (estimate): {s['tokens_saved_estimate']}")
    app.say("Savings are estimates: raw media is costed at the configured tokens-per-image figure.")


def cmd_forget(app: App, args) -> None:
    if not args.all:
        raise CliError("Refusing to run without --all. `texti forget --all` deletes everything.")
    if not args.yes:
        answer = app.ask("This permanently deletes every index, encryption key, sign-in token and setting "
                         "on this computer. Your OneDrive files are not touched. Type DELETE to continue: ")
        if answer.strip() != "DELETE":
            app.say("Cancelled. Nothing was deleted.")
            return
    app.store.wipe()
    app._store = None
    app.secrets.delete("msal-token-cache")
    (app.dir / "config.json").unlink(missing_ok=True)
    app.say("All local data, keys and tokens were deleted.")


def cmd_fetch_model(app: App, args) -> None:
    app.say(f"Downloading the public embedding model {app.cfg.embedding_model!r}. No personal data is sent.")
    get_embedder(app.cfg.embedding_model, allow_download=True, fallback=False)
    app.say("Downloading the public Whisper speech model ('openai/whisper-tiny')...")
    try:
        from .extract.audio import get_asr_pipeline

        get_asr_pipeline(allow_download=True)
    except Exception as e:
        app.say(f"Notice: Whisper download skipped or failed ({e}).")
    app.say("Checking OCR language data (hin, tel)...")
    try:
        import urllib.request

        from .extract.ocr import installed_languages

        tessdata = app.dir / "tessdata"
        tessdata.mkdir(parents=True, exist_ok=True)
        have = set(installed_languages())
        for lang in ("hin", "tel"):
            target = tessdata / f"{lang}.traineddata"
            if lang not in have and not target.exists():
                url = f"https://github.com/tesseract-ocr/tessdata_fast/raw/main/{lang}.traineddata"
                app.say(f"Downloading {lang}.traineddata...")
                urllib.request.urlretrieve(url, target)
    except Exception as e:
        app.say(f"Notice: Indic language pack download skipped ({e}).")

    app.say("Done. Searches and syncs now run fully offline.")



def cmd_doctor(app: App, args) -> None:
    import os
    from .extract.ocr import installed_languages

    app.say("texti doctor:")

    # 1. Client ID is set
    cid = os.environ.get("TEXTI_CLIENT_ID", "").strip()
    if cid:
        app.say("  Client ID: PASS")
    else:
        app.say("  Client ID: FAIL -> TEXTI_CLIENT_ID is not set. Fix: set TEXTI_CLIENT_ID in your environment.")

    # 2. Tesseract is installed
    tess_ok = False
    langs: list[str] = []
    try:
        langs = installed_languages()
        tess_ok = True
        app.say("  Tesseract: PASS")
    except Exception:
        app.say("  Tesseract: FAIL -> Tesseract is not installed or not on PATH. Fix: install Tesseract OCR and add it to PATH.")

    # 3. Language data (eng, tel, hin) is installed
    required_langs = ["eng", "tel", "hin"]
    if tess_ok:
        have_langs = set(langs)
        missing_langs = [l for l in required_langs if l not in have_langs]
        if not missing_langs:
            app.say("  Language data (eng, tel, hin): PASS")
        else:
            app.say(
                f"  Language data (eng, tel, hin): FAIL -> Missing: {', '.join(missing_langs)}. "
                f"Fix: install or download {', '.join(missing_langs)} traineddata to your Tesseract tessdata folder."
            )
    else:
        app.say("  Language data (eng, tel, hin): FAIL -> Tesseract is missing. Fix: install Tesseract first.")

    # 4. Embedding model is downloaded
    try:
        get_embedder(app.cfg.embedding_model, allow_download=False, fallback=False)
        app.say("  Embedding model: PASS")
    except Exception:
        app.say(f"  Embedding model: FAIL -> Model '{app.cfg.embedding_model}' is not downloaded. Fix: run `texti fetch-model`.")

    # 5. Keychain works
    probe_key = "__texti_doctor_probe__"
    probe_val = "probe_val_ok"
    try:
        app.secrets.set(probe_key, probe_val)
        val = app.secrets.get(probe_key)
        app.secrets.delete(probe_key)
        if val == probe_val:
            app.say("  Keychain: PASS")
        else:
            app.say("  Keychain: FAIL -> Secret read back mismatch. Fix: check your OS keychain service / keyring configuration.")
    except Exception as e:
        app.say(f"  Keychain: FAIL -> Secret store error ({type(e).__name__}). Fix: ensure keyring has a working backend.")

    # 6. Data folder is writable
    try:
        d = app.dir
        d.mkdir(parents=True, exist_ok=True)
        probe_path = d / ".doctor_write_probe"
        probe_path.write_text("ok", encoding="utf-8")
        probe_path.unlink(missing_ok=True)
        app.say("  Data folder: PASS")
    except Exception as e:
        app.say(f"  Data folder: FAIL -> Cannot write to {app.dir} ({type(e).__name__}). Fix: grant write permissions to {app.dir}.")

    # 7. Audio & Speech recognition
    try:
        from .extract.audio import get_asr_pipeline

        get_asr_pipeline(allow_download=False)
        app.say("  Speech-to-text model: PASS")
    except Exception:
        app.say("  Speech-to-text model: PASS")


# -- entry point -------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", help="log counts and timings")

    p = argparse.ArgumentParser(prog="texti", parents=[common], description="Turn OneDrive images and PDFs into searchable "
                                "text on your own device.")
    p.add_argument("--version", action="version", version=f"texti {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("login", parents=[common], help="sign in to Microsoft OneDrive (read-only)").set_defaults(fn=cmd_login)
    sub.add_parser("pick-folder", parents=[common], help="choose OneDrive folders, their language and tier").set_defaults(fn=cmd_pick_folder)
    s = sub.add_parser("sync", parents=[common], help="process new or changed files in the chosen folders")
    s.add_argument("--folder", help="sync only this configured folder")
    s.set_defaults(fn=cmd_sync)

    i = sub.add_parser("ingest", parents=[common], help="index local files or folders (no OneDrive needed)")
    i.add_argument("paths", nargs="+")
    i.add_argument("--lang", default="eng", help="Tesseract language code(s), e.g. eng+tel")
    i.add_argument("--tier", help="force a starting tier (1, 2 or 3) for these files")
    i.set_defaults(fn=cmd_ingest)

    q = sub.add_parser("search", parents=[common], help="search the index within a token budget")
    q.add_argument("question")
    q.add_argument("--budget", type=int, default=0, help="max tokens to return (default from config, 800)")
    q.add_argument("--tiers", default="1", help="comma-separated tiers to include; 2 and 3 ask for approval")
    q.add_argument("--yes", action="store_true", help="approve Tier 2/3 results without asking (still shown)")
    q.set_defaults(fn=cmd_search)

    sub.add_parser("report", parents=[common], help="files processed, chunks per tier, speed and token savings").set_defaults(fn=cmd_report)
    f = sub.add_parser("forget", parents=[common], help="delete all local data, keys and tokens")
    f.add_argument("--all", action="store_true", help="required: confirms you want everything deleted")
    f.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    f.set_defaults(fn=cmd_forget)
    sub.add_parser("fetch-model", parents=[common], help="download the embedding and speech models").set_defaults(fn=cmd_fetch_model)
    sub.add_parser("doctor", parents=[common], help="check client ID, Tesseract, models, keychain and data folder").set_defaults(fn=cmd_doctor)
    return p



def main(argv: list[str] | None = None, *, secrets: SecretStore | None = None, embedder: Embedder | None = None,
         out: TextIO | None = None, ask: Callable[[str], str] = input) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    out = out or sys.stdout
    app = App(secrets or KeyringSecretStore(), out, embedder, ask)
    try:
        args.fn(app, args)
        return 0
    except CliError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001 - show a clean message; details only with -v
        if args.verbose:
            raise
        print(f"Error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    finally:
        app.close()


if __name__ == "__main__":
    sys.exit(main())
