# Texti 🔍

> **Local-first search for your OneDrive media** — turns images, PDFs, audio, and video into fully searchable, privacy-classified text, entirely on your own device.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://python.org)

[![Tests](https://img.shields.io/badge/tests-158%20passed-brightgreen)](#development)

---

## What is Texti?

Texti is a command-line tool that indexes your OneDrive files locally so you can search across your documents without uploading anything to a cloud AI. It:

- **Extracts text** from PDFs (native + scanned), images (OCR), and audio/video (speech-to-text)
- **Classifies privacy** automatically — Aadhaar, PAN, bank statements, medical records, and more
- **Stores everything encrypted** using AES-GCM with per-tier keys in your OS keychain
- **Searches fast** using a hybrid BM25 + semantic embedding approach
- **Never calls an external AI API** — all processing is local after a one-time model download

---

## Table of Contents

- [Privacy & Data Storage](#privacy--data-storage)
- [Privacy Tiers](#privacy-tiers)
- [Setup](#setup)
- [Commands](#commands)
- [How Search Works](#how-search-works)
- [Project Layout](#project-layout)
- [Development](#development)
- [Known Limits](#known-limits)

---

## Privacy & Data Storage

All your data stays on your machine. Here is exactly what is stored where:

| Data | Location | Protection |
|------|----------|------------|
| Extracted text, paths, embeddings | `~/.texti/data/tier{1,2,3}.db` | AES-GCM encrypted, one key per tier |
| Encryption keys + MS sign-in tokens | OS keychain (`keyring`) | OS-managed, never written to disk |
| Original files | OneDrive only | Downloaded to a temp folder, **deleted immediately** after processing |
| File bookkeeping (IDs, hashes, timestamps) | `~/.texti/data/meta.db` | No filenames, paths, or text stored |

**Network access is limited to:**

1. **Microsoft Graph** — sign-in, folder listing, and file download. Read-only (`Files.Read`).
2. **One-time model download** via `texti fetch-model` — pulls public model weights from Hugging Face. No personal data is ever sent.

> No telemetry. No external AI API. Log output contains only opaque IDs, counts, and timings — never file names or text.

---

## Privacy Tiers

Texti automatically assigns each page of content to one of three privacy tiers:

| Tier | Description | Search Behaviour | Sensitive Values Masked |
|------|-------------|-----------------|------------------------|
| **Tier 1** — General | Study notes, slides, public documents | Returned directly | Card numbers, OTPs, API keys, generic IDs |
| **Tier 2** — Personal | Photos, chats, personal notes *(default for unknown files)* | Requires explicit approval | Also: email addresses, phone numbers |
| **Tier 3** — Private | ID cards, bank statements, medical and legal documents | Requires approval + shows a warning | Also: Aadhaar, PAN, IFSC, account numbers |

### How tiers are assigned

- A **folder rule** sets the starting tier; files with no rule default to Tier 2.
- Content can only **raise** a tier — never lower it. An Aadhaar number found in a "Tier 1" folder makes that page Tier 3.
- Sensitive values are **masked before chunking, embedding, or storing**. For example, Aadhaar numbers are stored as `XXXX XXXX 1234`.

### Detectors included

| Detector | Method |
|----------|--------|
| Aadhaar number | Verhoeff checksum |
| PAN card | Regex pattern |
| Credit / debit card | Luhn algorithm |
| IFSC code | Regex + artifact filter |
| Indian passport | Regex pattern |
| Bank account number | Context-aware regex |
| OTP / PIN | Regex pattern |
| API keys & passwords | Regex heuristics |
| Phone number, email | Regex pattern |
| Document-type phrases | `"bank statement"`, `"prescription"`, `"medical report"` -> Tier 3 |
| Identity wording | `"Aadhaar"`, `"government-issued"`, `"employee id"` -> Tier 3 |

---

## Setup

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.11+ | [python.org](https://www.python.org) |
| Tesseract OCR | Install language packs for your content (e.g. `tesseract-ocr-hin`, `tesseract-ocr-tel`) |
| Microsoft Entra app | Public client, redirect URI `http://localhost`, delegated permission `Files.Read` |

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/nithish-74/Texti.git
cd Texti

# 2. Install in editable mode (creates the `texti` command)
pip install -e .

# 3. Set your Microsoft app client ID
export TEXTI_CLIENT_ID="your-client-id-here"
# On Windows (PowerShell):
# $env:TEXTI_CLIENT_ID = "your-client-id-here"

# 4. Download the embedding model (one-time, ~90 MB)
texti fetch-model

# 5. Verify your setup
texti doctor
```

> **Note:** If you skip step 4, Texti falls back to a character-based embedder and warns you. Search still works, but quality is significantly lower.

---

## Commands

```
texti [-v] <command> [options]
```

| Command | Description |
|---------|-------------|
| `texti login` | Sign in to OneDrive via your browser; tokens are saved to the OS keychain |
| `texti pick-folder` | Choose OneDrive folders, set the OCR language (e.g. `eng+tel`) and privacy tier |
| `texti sync [--folder NAME]` | Process new or changed files; uses Graph delta API where available |
| `texti ingest PATH...` | Index local files or folders directly (no OneDrive required) |
| `texti search "<question>"` | Search indexed content within a token budget |
| `texti report` | Show file counts per tier, chunk stats, processing speed, and token savings |
| `texti forget --all` | Wipe all indexes, keys, tokens, and settings from this machine |
| `texti fetch-model` | Download the semantic embedding model from Hugging Face |
| `texti doctor` | Check system readiness: client ID, Tesseract, language packs, models, keychain |

### Search options

```bash
texti search "invoice from March" --budget 800 --tiers 1
```

| Option | Default | Description |
|--------|---------|-------------|
| `--budget N` | 800 | Maximum tokens to return |
| `--tiers N [N...]` | 1 2 3 | Tiers to include in results |

Search returns **one-line summaries** for the best matching files first, then **full chunk text** until the token budget is consumed. Low-confidence OCR results are flagged so you know to verify the original file.

---

## How Search Works

Texti uses a **hybrid retrieval** strategy that combines two approaches:

```
Query
  |-- BM25 keyword scoring   (exact matches: invoice numbers, IDs, code)
  `-- Dense vector search    (semantic matches: "money transfer" ~ "wire payment")
         |-- all-MiniLM-L6-v2  (English)
         `-- paraphrase-multilingual-MiniLM-L12-v2  (Hindi / Telugu / others)
                |
                v
         Fused rank score -> top-K chunks -> token budget filter -> results
```

The `hybrid_weight` config value controls the BM25 vs. dense split (default: `0.5`, equal weight).

---

## Project Layout

```
Texti/
+-- src/texti/
|   +-- cli.py                  # Entry point - all subcommands
|   +-- config.py               # Config loading and defaults
|   +-- pipeline.py             # Ingest orchestration
|   +-- secrets_store.py        # Keychain read/write helpers
|   +-- tokens.py               # Token counting and budget management
|   |
|   +-- onedrive/
|   |   +-- auth.py             # MSAL device-code + token refresh
|   |   +-- graph.py            # Microsoft Graph API client
|   |   `-- sync.py             # Delta sync and file download
|   |
|   +-- extract/
|   |   +-- ocr.py              # Tesseract OCR (multi-language)
|   |   +-- pdf.py              # Native PDF text extraction (pdfminer)
|   |   `-- audio.py            # Whisper speech-to-text via PyAV
|   |
|   +-- privacy/
|   |   +-- patterns.py         # Regex + checksum detectors
|   |   +-- classifier.py       # Tier assignment logic
|   |   `-- redact.py           # Masking before storage
|   |
|   +-- index/
|   |   +-- chunker.py          # Text splitting
|   |   +-- embedder.py         # SentenceTransformer / fallback embedder
|   |   `-- store.py            # AES-GCM encrypted SQLite store
|   |
|   `-- retrieve/
|       +-- bm25.py             # BM25 keyword ranking
|       +-- search.py           # Hybrid BM25 + dense search
|       `-- budget.py           # Token budget enforcement
|
`-- tests/                      # 158 unit + integration tests (all in-memory)
```

---

## Development

```bash
# Run tests (no extra packages needed)
python -m unittest discover -s tests -t .

# Run with pytest (after: pip install -e .[dev])
pytest

# Run with verbose output
pytest -v
```

All tests use synthetic in-memory data — no real files, no network, no keychain.

---

## Known Limits

- **OCR accuracy**: Misread characters can cause detectors to miss sensitive values. Folder rules and identity-wording detection act as a second line of defence, but review your tier assignments manually.
- **Processing speed**: OCR and speech transcription are compute-intensive. A GPU helps significantly for large batches.
- **OneDrive for Business delta**: Folder-level delta is not available on all drives. When unavailable, the full folder is scanned on each sync. Unchanged files are skipped by hash (no re-download), but deletions are not detected until the next full scan.
- **Whisper model size**: `texti fetch-model` downloads `whisper-tiny` by default. Accuracy improves with larger models (`base`, `small`), which can be configured manually.

---

## License

---

*Built as a demonstration of a privacy-preserving, fully local media-to-text pipeline.*
