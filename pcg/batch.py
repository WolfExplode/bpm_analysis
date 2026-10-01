"""Run queue: Recordings in, Analyses (in the library) out. Shared by the run screen and the CLI.

Also owns the filename conventions: a start-BPM hint read from a `[start,min-maxbpm]`
tag, and writing that tag into input filenames (an explicit action, never automatic).
"""
from __future__ import annotations

import logging
import os
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from pcg import analysis, recording

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunSettings:
    """Run-wide choices (run screen / CLI). Engine tuning lives only in config.py."""
    springer: bool = False
    auto_switch: bool = False
    channel: str = recording.CHANNEL_MIXED
    jobs: int = 1

    def engine_settings(self, start_sec: float = 0.0) -> Dict:
        return {
            "use_springer_algorithm": self.springer,
            "auto_switch_algorithm": self.auto_switch,
            "analysis_start_sec": float(start_sec),
        }

    @property
    def effective_jobs(self) -> int:
        # The Springer HSMM's memory grows steeply with duration (several GB per minute of
        # high-rate audio), so anything that may run it is never parallelised.
        if self.springer or self.auto_switch:
            return 1
        return max(1, int(self.jobs))


@dataclass
class Job:
    path: str
    start_bpm_hint: Optional[float] = None   # None = from filename (if tagged)
    start_sec: float = 0.0

    def hint(self) -> Optional[float]:
        return self.start_bpm_hint if self.start_bpm_hint is not None else start_bpm_from_filename(self.path)


@dataclass
class JobResult:
    path: str
    ok: bool
    analyses: List[str] = field(default_factory=list)   # library paths
    bpm: List[Optional[Dict]] = field(default_factory=list)  # bpm summary per analysis
    gate_failed: List[bool] = field(default_factory=list)
    gate_reasons: List[List[str]] = field(default_factory=list)
    error: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Filename BPM tags
# ─────────────────────────────────────────────────────────────────────────────

_TAG_TAIL_RE = re.compile(
    r"(?:\s+(?:"
    r"\[\s*\d+\s*,\s*\d+\s*-\s*\d+\s*bpm\s*\]"
    r"|\[\s*\d+\s*to\s*\d+\s*bpm\s*\]"
    r"|\[\s*\d+\s*bpm\s*\]"
    r"|\d+\s*,\s*\d+\s*-\s*\d+\s*bpm"
    r"|\d+\s*to\s*\d+\s*bpm"
    r"|\d+\s*bpm"
    r"))+$",
    re.IGNORECASE,
)
_MAX_BASENAME_LEN = 255
_MAX_FULL_PATH_LEN = 260


def start_bpm_from_filename(path: os.PathLike | str) -> Optional[float]:
    """Start BPM from the rightmost tag: '[120,60-150bpm]', '90to132bpm' or '150bpm'."""
    base = os.path.basename(str(path))
    for pattern in (r"(\d+)\s*,\s*\d+\s*-\s*\d+\s*bpm", r"(\d+)\s*to\s*\d+\s*bpm", r"(\d+)\s*bpm"):
        matches = list(re.finditer(pattern, base, flags=re.IGNORECASE))
        if matches:
            return float(matches[-1].group(1))
    return None


def strip_bpm_tags(stem: str) -> str:
    return _TAG_TAIL_RE.sub("", stem).rstrip()


def format_bpm_tag(start_bpm: float, min_bpm: float, max_bpm: float) -> str:
    return f"[{int(round(start_bpm))},{int(round(min_bpm))}-{int(round(max_bpm))}bpm]"


def renamed_with_bpm(path: os.PathLike | str, bpm: Dict) -> Path:
    """Target path with the trailing BPM tag replaced (does not touch the disk)."""
    p = Path(path)
    clean = strip_bpm_tags(p.stem)
    tag = format_bpm_tag(bpm["start_bpm"], bpm["min_bpm"], bpm["max_bpm"])
    return p.with_name((f"{clean} {tag}" if clean else tag) + p.suffix)


def rename_with_bpm(path: os.PathLike | str, bpm: Optional[Dict]) -> Tuple[Optional[Path], str]:
    """Rename a recording to carry its BPM tag. Returns (new path or None, reason if skipped).

    A matching Annotation sidecar is renamed along with it, so it stays easy to find
    (it is matched by fingerprint either way).
    """
    if not bpm:
        return None, "no BPM result"
    src = Path(path)
    dst = renamed_with_bpm(src, bpm)
    if len(dst.name) > _MAX_BASENAME_LEN or len(str(dst.resolve())) > _MAX_FULL_PATH_LEN:
        return None, "the new name would be too long"
    if os.path.normcase(str(src.resolve())) == os.path.normcase(str(dst.resolve())):
        return None, "already tagged"
    if dst.exists():
        return None, f"{dst.name} already exists"
    from pcg import annotation

    sidecar = annotation.sidecar_path(src)
    os.rename(src, dst)
    if sidecar.exists() and not annotation.sidecar_path(dst).exists():
        os.rename(sidecar, annotation.sidecar_path(dst))
    return dst, ""


# ─────────────────────────────────────────────────────────────────────────────
# Inputs
# ─────────────────────────────────────────────────────────────────────────────

_EXT_PREFERENCE = {".wav": 2, ".flac": 1, ".mp3": 1, ".m4a": 1, ".ogg": 1, ".mp4": 1, ".mkv": 1, ".mov": 1}


def _dedupe_key(path: Path) -> str:
    stem = "".join(c for c in path.stem if unicodedata.category(c)[0] in "LNPZ")
    return " ".join(stem.split()).casefold()


def collect(paths: Iterable[os.PathLike | str]) -> List[Path]:
    """Audio files under the given files/folders; when the same name exists as WAV and
    as a compressed file, keep the WAV."""
    best: Dict[Tuple[str, str], Tuple[int, Path]] = {}
    order: List[Tuple[str, str]] = []
    for p in recording.find_recordings(list(paths)):
        key = (os.path.normcase(str(p.parent.resolve())), _dedupe_key(p))
        score = _EXT_PREFERENCE.get(p.suffix.lower(), 0)
        if key not in best:
            order.append(key)
            best[key] = (score, p)
        elif score > best[key][0]:
            best[key] = (score, p)
    return [best[k][1] for k in order]


# ─────────────────────────────────────────────────────────────────────────────
# Running
# ─────────────────────────────────────────────────────────────────────────────

ProgressFn = Callable[[str, str], None]  # (recording path, message)


def analyze(job: Job, settings: RunSettings, library_root: str, fingerprint: str,
            progress: Optional[Callable[[str], None]] = None) -> JobResult:
    """Analyze one Recording and store its Analyses in the library. Never raises."""
    try:
        lib = analysis.Library(library_root)
        built = analysis.build(
            job.path, fingerprint=fingerprint, channel=settings.channel,
            run_settings=settings.engine_settings(job.start_sec), start_bpm_hint=job.hint(),
            progress=progress,
        )
        res = JobResult(job.path, True)
        for a in built:
            res.analyses.append(str(lib.save(a)))
            res.bpm.append(a.summary.get("bpm"))
            gate = a.summary.get("gate") or {}
            res.gate_failed.append(bool(gate.get("failed")))
            res.gate_reasons.append(list(gate.get("reasons") or []))
        return res
    except Exception as e:  # one bad file must not stop the queue
        log.exception("analysis failed: %s", job.path)
        return JobResult(job.path, False, error=f"{type(e).__name__}: {e}")


def _analyze_in_worker(job_dict: Dict, settings_dict: Dict, library_root: str, fingerprint: str) -> JobResult:
    logging.basicConfig(level=logging.WARNING)
    return analyze(Job(**job_dict), RunSettings(**settings_dict), library_root, fingerprint)


def run(jobs: List[Job], settings: RunSettings, library: analysis.Library,
        on_progress: Optional[ProgressFn] = None,
        on_done: Optional[Callable[[JobResult], None]] = None) -> List[JobResult]:
    """Run a queue. Fingerprints are computed here (single process, shared cache)."""
    results: List[JobResult] = []
    prints = {}
    for job in jobs:
        try:
            prints[job.path] = library.fingerprints.get(job.path)
        except Exception as e:
            r = JobResult(job.path, False, error=f"cannot read audio: {e}")
            results.append(r)
            if on_done:
                on_done(r)
    todo = [j for j in jobs if j.path in prints]
    n = settings.effective_jobs
    if n <= 1 or len(todo) <= 1:
        for job in todo:
            r = analyze(job, settings, str(library.root), prints[job.path],
                        progress=(lambda m, p=job.path: on_progress(p, m)) if on_progress else None)
            results.append(r)
            if on_done:
                on_done(r)
        return results
    with ProcessPoolExecutor(max_workers=n) as pool:
        futures = {
            pool.submit(_analyze_in_worker, asdict(job), asdict(settings), str(library.root), prints[job.path]): job
            for job in todo
        }
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            if on_done:
                on_done(r)
    return results
