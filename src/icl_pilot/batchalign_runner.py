from __future__ import annotations

from pathlib import Path
import shutil
import subprocess


def resolve_batchalign_executable() -> str:
    project_local = Path(".venv/bin/batchalign")
    if project_local.exists():
        return str(project_local)

    on_path = shutil.which("batchalign")
    if on_path:
        return on_path

    raise FileNotFoundError("Could not find a batchalign executable.")


def build_morphotag_command(
    input_dir: str,
    output_dir: str,
    keeptokens: bool = True,
    lexicon: str | None = None,
    override_cache: bool = False,
    force_cpu: bool = False,
    workers: int | None = None,
) -> list[str]:
    cmd = [resolve_batchalign_executable()]
    if force_cpu:
        cmd.append("--force-cpu")
    if workers is not None:
        cmd.extend(["--workers", str(workers)])

    cmd.extend(["morphotag"])
    cmd.append("--keeptokens" if keeptokens else "--retokenize")
    cmd.append("--override-cache" if override_cache else "--use-cache")

    if lexicon:
        cmd.extend(["--lexicon", lexicon])

    cmd.extend([input_dir, output_dir])
    return cmd


def run_morphotag(
    input_dir: str,
    output_dir: str,
    keeptokens: bool = True,
    lexicon: str | None = None,
    override_cache: bool = False,
    force_cpu: bool = False,
    workers: int | None = None,
    dry_run: bool = False,
) -> int:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cmd = build_morphotag_command(
        input_dir=input_dir,
        output_dir=output_dir,
        keeptokens=keeptokens,
        lexicon=lexicon,
        override_cache=override_cache,
        force_cpu=force_cpu,
        workers=workers,
    )

    print("Command:")
    print(" ".join(cmd))

    if dry_run:
        return 0

    completed = subprocess.run(cmd, check=False)
    return completed.returncode
