## TTS Batch Synthesizer

This workspace includes a batch Text-to-Speech tool for patient result narratives.

- Entry file: [src/tts.py](src/tts.py)
- Optional normalizer: [src/text_normalizer.py](src/text_normalizer.py)
- Optional helpers: [src/tts_extras.py](src/tts_extras.py)

### Dependencies

Install the Coqui TTS library (and optional extras) in your Python environment:

```bash
pip install TTS
```

If you see import errors for `requests` or other libraries in unrelated scripts, install them as needed:

```bash
pip install requests
```

### Running (Windows)

- Simple run (script mode, original behavior):

```bash
python src/tts.py --input-json <path/to/patient_results.json> --speaker-wav <path/to/speaker.wav> --output-dir <out/dir>
```

- Module mode (enables optional relative imports):

```bash
python -m src.tts --input-json <path/to/patient_results.json> --speaker-wav <path/to/speaker.wav> --output-dir <out/dir>
```

### Optional Flags

- `--dry-run`: Process text and log actions without generating audio
- `--resume`: Skip items when target wav already exists
- `--filename-field`: Use a JSON field to derive output filename (sanitized)
- `--write-metadata`: Write a sidecar JSON per item with synthesis details
- `--strip-newlines`: Convert newlines to spaces during normalization
- `--no-acronym`: Disable acronym expansion (e.g., ICH → intracerebral hemorrhage)
- `--retries N`: Retry synthesis up to N times on transient errors
- `--sleep-seconds S`: Sleep S seconds between items
- `--hooks-log PATH`: Emit per-item telemetry as JSONL (optional)
- `--run-summary PATH`: Write aggregate run summary JSON (optional)
- `--max-failures N`: Abort the batch after N failures if desired

Optional observability and guardrails live in [src/tts_optional_features.py](src/tts_optional_features.py); if absent, the core pipeline behaves exactly as before.

### Notes

- GPU vs CPU: pass `--cpu` to force CPU; otherwise GPU is used if available.