"""Recording metadata sidecars in the analysis library, keyed by audio identity."""
import json
import os
from pathlib import Path


def metadata_path(library_root, fingerprint):
    # Fingerprints normally are hexadecimal; also support readable test identities.
    if not fingerprint or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                              for c in fingerprint):
        raise ValueError("Invalid recording fingerprint")
    return Path(library_root) / f"{fingerprint}.recording.json"


def load(library_root, fingerprint):
    path = metadata_path(library_root, fingerprint)
    if not path.exists():
        return {"version": 1, "person": {}, "recoveries": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or data.get("version") != 1
            or not isinstance(data.get("person"), dict) or not isinstance(data.get("recoveries"), dict)
            or any(not isinstance(value, dict) for value in data.get("recoveries", {}).values())):
        raise ValueError("Unsupported recording metadata format")
    return data


def save(library_root, fingerprint, data):
    path = metadata_path(library_root, fingerprint)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)
