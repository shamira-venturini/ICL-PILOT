# Local CHAT validator

`chatter` is the previously verified Chatter 0.30.0 executable, relocated from
the temporary conversion audit without changing its bytes. It is ignored by
Git. [`source.json`](source.json) records its release URL, original platform,
version output, download hash and executable hash.

From the repository root:

```bash
tools/chatter/chatter validate PATH_TO_CHAT_FILES --force --strict-linkers --format json
```

To include the native validator in the formatting regression tests, set
`ICL_PILOT_CHATTER` to the absolute path of this executable before running
`tests/test_generated_chat.py`. Other platforms need their corresponding
Chatter executable; this local copy is the recorded macOS x86-64 build.

Historical validation reports are under `docs/provenance/`; the current
generated-transcript reports are under `data/qc/`. The full original tool
download and audit tree remain in the verified external outputs archive.
