from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class ArtifactReference:
    artifact_id: str
    sha256: str
    size_bytes: int
    media_type: str
    relative_path: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "ArtifactReference":
        return cls(**value)


class LocalArtifactStore:
    """Content-addressed local storage for large trace payloads."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def put_bytes(
        self,
        content: bytes,
        *,
        media_type: str = "application/octet-stream",
    ) -> ArtifactReference:
        digest = hashlib.sha256(content).hexdigest()
        relative = Path("sha256") / digest[:2] / digest
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{digest}.",
                dir=destination.parent,
            )
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_name, destination)
            finally:
                temporary = Path(temporary_name)
                if temporary.exists():
                    temporary.unlink()
        return ArtifactReference(
            artifact_id=f"artifact_{digest}",
            sha256=digest,
            size_bytes=len(content),
            media_type=media_type,
            relative_path=relative.as_posix(),
        )

    def put_text(
        self,
        content: str,
        *,
        media_type: str = "text/plain; charset=utf-8",
    ) -> ArtifactReference:
        return self.put_bytes(content.encode("utf-8"), media_type=media_type)

    def read_bytes(self, reference: ArtifactReference) -> bytes:
        path = self.root / reference.relative_path
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != reference.sha256 or len(content) != reference.size_bytes:
            raise ValueError(f"Artifact integrity check failed: {reference.artifact_id}")
        return content

    def read_text(self, reference: ArtifactReference) -> str:
        return self.read_bytes(reference).decode("utf-8")

