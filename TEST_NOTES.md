# TEST_NOTES.md

## Real-File Ingestion & Search Evaluation Notes

### 1. Crashes & CLI Argument Issues
- **`--folder-tier` Argument Rejection**: Running `texti ingest real_test/ --folder-tier 2` crashed with:
  `texti: error: unrecognized arguments: --folder-tier 2`. The supported CLI flag is `--tier`.
- **Global Flag Placement**: Placing `-v` after the subcommand (e.g. `texti ingest real_test/ --tier 2 -v`) resulted in `unrecognized arguments: -v`.
  - **Status: RESOLVED.** Common parent parser added with `-v`/`--verbose` flag across all subcommands.
- **Config Deserialization Failure on Legacy Fields**: When `config.json` contained `folder_id` in `folders`, `Config.load()` raised `TypeError: FolderConfig.__init__() got an unexpected keyword argument 'folder_id'`.
  - **Status: RESOLVED** by adding `_folder_from_raw()` in `config.py`.

### 2. Tier Classification Issues
- **Sensitive ID Card Classified as Tier 2 instead of Tier 3**:
  - File ID: `local-ade48a0ee8ce1665313c`
  - Observation: Landed in Tier 2 despite being an identity document. Because the synthetic ID pattern (`TEST-0000-1234`) did not match exact Aadhaar/PAN regexes, it was not automatically escalated to Tier 3.
  - **Status: RESOLVED.** Expanded `PAGE_PRIVATE_PHRASES` with generic and synthetic ID markers (`"id number"`, `"id no."`, `"employee id"`, `"student id"`, `"government-issued"`), properly escalating all identity documents to Tier 3.
- **Accidental Redaction False Positive**:
  - File ID: `local-8a046f95ed5e74b6b883`
  - Observation: When non-English text was OCR'd with the English model, garbled text produced character combinations that falsely triggered `[REDACTED:IFSC]`.
  - **Status: RESOLVED.** Implemented `is_valid_ifsc()` checking bank structure and rejecting repetitive OCR noise (e.g. `OOOO0OOOOOO`).

### 3. OCR Text & Language Quality Issues
- **Missing Non-English Language Models**:
  - File ID: `local-8a046f95ed5e74b6b883` (Telugu notes): Ingested with default `eng` model. Non-Latin characters transcribed as sequences of letters ('O', '0', 'C'), dropping confidence to `0.38`.
  - File ID: `local-e6fc805816816427f5e0` (Hindi notes): Ingested with default `eng` model. Distorted Latin letters, confidence `0.50`.
  - **Status: RESOLVED.** Downloaded `hin` and `tel` traineddata to `~/.texti/tessdata`, auto-configured `TESSDATA_PREFIX`, and enabled multi-language combination (`eng+hin+tel`). Telugu and Hindi notes now transcribe with native characters and confidence >= 0.63 (`PASS`).


### 4. Performance & Processing Latencies
- **Total Ingestion Time**: 22 files processed in 17.7s (average 0.80s per file).
- **Processing Time by File ID**:
  - `local-630fd7eb75e9f063b12d`: 1.34s (slowest - multi-page PDF processing)
  - `local-856ea5772b421d88c76d`: 0.53s (scanned PDF page rendering + OCR)
  - `local-fe227161480c78bd6d56`: 0.41s
  - `local-35a098a849ba07d0984b`: 0.41s
  - `local-8a046f95ed5e74b6b883`: 0.38s
  - `local-154dd99417e36aec3d4f`: 0.33s
  - `local-036878c2b33f170e6fa4`: 0.28s
  - `local-05406337af60ba73bb21`: 0.28s
  - `local-12_email`: `local-2db21c0e393774d8e38f`: 0.28s
  - `local-ade48a0ee8ce1665313c`: 0.28s
  - `local-be973ae3e449d5834c86`: 0.28s
  - `local-7f7907d172b884968bfb`: 0.28s
  - `local-885e629125cec4e60379`: 0.28s
  - `local-074dac7f1cdef09f1d1b`: 0.27s
  - `local-16766c117802a57fdbb5`: 0.27s
  - `local-21ad89892479f3212e5b`: 0.27s
  - `local-55f22615fa2f7d54164a`: 0.27s
  - `local-adf027044e43ac623001`: 0.27s
  - `local-e6fc805816816427f5e0`: 0.27s
  - `local-5703ae09297fc1ebf18c`: 0.26s
  - `local-fc7d5b6a290f1374bab4`: 0.26s
  - `local-4a8bce884a9d47b98003`: 0.02s (fastest - native text layer, OCR skipped)
