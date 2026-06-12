from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, List


_SECRET_PATTERNS = {
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    "bearer_token": re.compile(r"\bBearer\s+[A-Za-z0-9._~-]{16,}\b", re.IGNORECASE),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "password_assignment": re.compile(r"\bpassword\s*[:=]\s*[^\s,;]{8,}", re.IGNORECASE),
}


def sensitive_findings(text: str) -> List[Dict[str, str]]:
    findings = []
    for kind, pattern in _SECRET_PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append({"kind": kind, "value": match.group(0)})
    return findings


def purge_expired_artifacts(root: str | Path, *, retention_seconds: float, now: float | None = None) -> List[str]:
    root_path = Path(root).resolve()
    cutoff = (time.time() if now is None else now) - retention_seconds
    removed = []
    for path in root_path.rglob("*"):
        if path.is_file() and path.stat().st_mtime < cutoff:
            relative = path.resolve().relative_to(root_path)
            path.unlink()
            removed.append(relative.as_posix())
    return removed
