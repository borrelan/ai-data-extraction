#!/usr/bin/env python3
"""Join inventory and immutable-snapshot evidence into one source ledger.

Inventory and snapshot commands remain responsible for observing their own
boundary.  This module is the narrow join owner consumed by later parsers and
builders: the immutable snapshot is authoritative for source bytes, while
older inventory observations are retained as drift evidence.  The ledger is
metadata-only and never records source paths or content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

SOURCE_MANIFEST_SCHEMA = "ai-data-extraction/source-manifest/v1"
SOURCE_MANIFEST_VERSION = "1.0.0"
INVENTORY_SCHEMA = "ai-data-extraction/source-inventory/v1"
SNAPSHOT_SCHEMA = "ai-data-extraction/source-snapshot/v1"
ADMISSIBLE_SNAPSHOT_STATUSES = frozenset(
    {"bound", "snapshot_authoritative_changed_since_inventory"}
)
SOURCE_CLASS_ALIASES = {
    "conversation_database": "conversation_store",
    "conversation_storage": "conversation_store",
    "prime_session_active": "session_active",
    "prime_subagent_session": "subagent_session",
    "pi_session_active": "session_active",
    "pi_session_backup": "session_backup",
    "pi_advisor_overlay": "advisor_overlay",
    "pi_advisor_overlay_backup": "advisor_overlay",
}
PROVIDER_ALIASES = {
    "claude-code": "claude",
    "gemini-cli": "gemini",
    "opencode-cli": "opencode",
}
ROOT_LABEL_ALIASES = {
    "backup": "backups",
    "session-artifacts": "primary",
}


class SourceManifestError(ValueError):
    """Raised when source-boundary evidence cannot be joined safely."""


class SourceAdmissionError(SourceManifestError):
    """Raised when a source file cannot be admitted from the ledger."""


@dataclass(frozen=True)
class SourceRef:
    """One candidate source identity with snapshot-authoritative bytes."""

    provider: str
    root_label: str
    source_class: str
    treatment: str
    relative_path_sha256: str
    snapshot_relative_path_sha256: str | None
    source_sha256: str | None
    source_bytes: int | None
    snapshot_status: str
    inventory_sha256: str | None
    inventory_bytes: int | None
    inventory_status: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest_value(value: Any) -> str:
    return _digest_bytes(_canonical_json(value).encode("utf-8"))


def canonical_source_class(value: str) -> str:
    """Map legacy provider-specific labels to the ledger taxonomy."""

    return SOURCE_CLASS_ALIASES.get(value, value)


def canonical_provider(value: str) -> str:
    """Map adapter-specific provider labels to inventory labels."""

    return PROVIDER_ALIASES.get(value, value)


def canonical_root_label(value: str) -> str:
    """Map legacy adapter root labels to inventory/snapshot labels."""

    return ROOT_LABEL_ALIASES.get(value, value)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceManifestError(f"cannot load manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SourceManifestError(f"manifest must be an object: {path}")
    return value


def _require_schema(manifest: dict[str, Any], schema: str, path: Path) -> None:
    if manifest.get("schema_version") != schema:
        raise SourceManifestError(
            f"unsupported schema in {path}: {manifest.get('schema_version')!r}"
        )
    if not isinstance(manifest.get("files"), list):
        raise SourceManifestError(f"files must be a list in {path}")


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SourceManifestError(f"{field} must be a non-empty string")
    return value


def _optional_digest(value: Any, field: str) -> str | None:
    if value is None:
        return None
    value = _require_text(value, field)
    digest = value.removeprefix("sha256:")
    if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
        raise SourceManifestError(f"{field} must be a SHA-256 digest")
    return digest.lower()


def _optional_bytes(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or value < 0:
        raise SourceManifestError(f"{field} must be a non-negative integer")
    return value


def _key(entry: dict[str, Any], index: int, *, owner: str) -> tuple[str, str, str]:
    provider = _require_text(entry.get("provider"), f"{owner}[{index}].provider")
    root_label = _require_text(entry.get("root_label"), f"{owner}[{index}].root_label")
    relative = _optional_digest(
        entry.get("relative_path_sha256"),
        f"{owner}[{index}].relative_path_sha256",
    )
    if relative is None:
        raise SourceManifestError(f"{owner}[{index}].relative_path_sha256 is required")
    return provider, root_label, relative


def _index_entries(
    entries: list[Any], *, owner: str, predicate: str | None = None
) -> dict[tuple[str, str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for index, raw in enumerate(entries):
        if not isinstance(raw, dict):
            raise SourceManifestError(f"{owner}[{index}] must be an object")
        if predicate is not None and raw.get("treatment") != predicate:
            continue
        key = _key(raw, index, owner=owner)
        if key in indexed:
            raise SourceManifestError(f"duplicate source identity in {owner}: {key}")
        indexed[key] = raw
    return indexed


def _snapshot_ref(
    key: tuple[str, str, str],
    snapshot: dict[str, Any] | None,
    inventory: dict[str, Any] | None,
) -> SourceRef:
    provider, root_label, _relative = key
    if snapshot is not None:
        source_class = _require_text(snapshot.get("source_class"), "snapshot.source_class")
        treatment = _require_text(snapshot.get("treatment"), "snapshot.treatment")
        source_sha256 = _optional_digest(snapshot.get("sha256"), "snapshot.sha256")
        if source_sha256 is None:
            raise SourceManifestError("stable snapshot entry is missing sha256")
        source_bytes = _optional_bytes(snapshot.get("bytes"), "snapshot.bytes")
        if source_bytes is None:
            raise SourceManifestError("stable snapshot entry is missing bytes")
        snapshot_status = "bound"
        snapshot_relative = _optional_digest(
            snapshot.get("snapshot_relative_path_sha256"),
            "snapshot.snapshot_relative_path_sha256",
        )
    elif inventory is not None:
        source_class = _require_text(inventory.get("source_class"), "inventory.source_class")
        treatment = _require_text(inventory.get("treatment"), "inventory.treatment")
        source_sha256 = None
        source_bytes = None
        snapshot_status = "inventory_only"
        snapshot_relative = None
    else:
        raise AssertionError("source ref requires one side")

    inventory_sha256 = _optional_digest(
        inventory.get("file_sha256") if inventory is not None else None,
        "inventory.file_sha256",
    )
    inventory_bytes = _optional_bytes(
        inventory.get("bytes") if inventory is not None else None,
        "inventory.bytes",
    )
    inventory_status = "unhashed" if inventory is not None and inventory_sha256 is None else (
        "observed" if inventory is not None else "missing"
    )

    if snapshot is not None and inventory is not None:
        if inventory_sha256 == source_sha256 and inventory_bytes == source_bytes:
            snapshot_status = "bound"
        else:
            snapshot_status = "snapshot_authoritative_changed_since_inventory"
        if inventory.get("source_class") != snapshot.get("source_class"):
            snapshot_status = "source_class_drift"
        if inventory.get("treatment") != snapshot.get("treatment"):
            snapshot_status = "treatment_drift"

    return SourceRef(
        provider=provider,
        root_label=root_label,
        source_class=source_class,
        treatment=treatment,
        relative_path_sha256=key[2],
        snapshot_relative_path_sha256=snapshot_relative,
        source_sha256=source_sha256,
        source_bytes=source_bytes,
        snapshot_status=snapshot_status,
        inventory_sha256=inventory_sha256,
        inventory_bytes=inventory_bytes,
        inventory_status=inventory_status,
    )


def build_source_manifest(inventory_path: Path, snapshot_path: Path) -> dict[str, Any]:
    """Join candidate inventory evidence to stable snapshot evidence."""

    inventory_path = Path(inventory_path)
    snapshot_path = Path(snapshot_path)
    inventory = _load_json(inventory_path)
    snapshot = _load_json(snapshot_path)
    _require_schema(inventory, INVENTORY_SCHEMA, inventory_path)
    _require_schema(snapshot, SNAPSHOT_SCHEMA, snapshot_path)

    inventory_entries = _index_entries(
        inventory["files"], owner="inventory.files", predicate="candidate"
    )
    snapshot_entries = _index_entries(
        [
            entry
            for entry in snapshot["files"]
            if isinstance(entry, dict) and entry.get("status") == "stable"
        ],
        owner="snapshot.files",
    )

    refs: list[SourceRef] = []
    for key in sorted(set(inventory_entries) | set(snapshot_entries)):
        refs.append(_snapshot_ref(key, snapshot_entries.get(key), inventory_entries.get(key)))

    counts = {
        "inventory_candidate_files": len(inventory_entries),
        "snapshot_stable_files": len(snapshot_entries),
        "joined_files": sum(
            ref.inventory_status != "missing" and ref.snapshot_status != "inventory_only"
            for ref in refs
        ),
        "bound_files": sum(ref.snapshot_status == "bound" for ref in refs),
        "changed_since_inventory": sum(
            ref.snapshot_status == "snapshot_authoritative_changed_since_inventory"
            for ref in refs
        ),
        "source_class_drift": sum(ref.snapshot_status == "source_class_drift" for ref in refs),
        "treatment_drift": sum(ref.snapshot_status == "treatment_drift" for ref in refs),
        "inventory_only": sum(ref.snapshot_status == "inventory_only" for ref in refs),
        "snapshot_only": sum(
            ref.inventory_status == "missing" and ref.snapshot_status != "inventory_only"
            for ref in refs
        ),
    }
    counts["unaccounted_files"] = counts["inventory_only"] + counts["snapshot_only"]
    snapshot_revision = _require_text(snapshot.get("snapshot_revision"), "snapshot_revision")
    file_rows: list[dict[str, Any]] = []
    for ref in refs:
        row = ref.as_dict()
        row["source_ref_sha256"] = "sha256:" + _digest_value(row)
        file_rows.append(row)
    manifest: dict[str, Any] = {
        "schema_version": SOURCE_MANIFEST_SCHEMA,
        "source_manifest_version": SOURCE_MANIFEST_VERSION,
        "privacy": "metadata_only_no_source_content_or_paths",
        "authority": "stable_snapshot",
        "inventory": {
            "schema_version": inventory["schema_version"],
            "artifact_sha256": _file_digest(inventory_path),
            "candidate_files": len(inventory_entries),
        },
        "snapshot": {
            "schema_version": snapshot["schema_version"],
            "artifact_sha256": _file_digest(snapshot_path),
            "snapshot_revision": snapshot_revision,
            "stable_files": len(snapshot_entries),
        },
        "counts": counts,
        "files": file_rows,
    }
    manifest["source_manifest_revision"] = "sha256:" + _digest_value(manifest)
    return manifest


def _verify_manifest_revision(manifest: Mapping[str, Any]) -> str:
    value = manifest.get("source_manifest_revision")
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise SourceAdmissionError("source manifest revision is missing")
    expected = "sha256:" + _digest_value(
        {key: item for key, item in manifest.items() if key != "source_manifest_revision"}
    )
    if value.lower() != expected.lower():
        raise SourceAdmissionError("source manifest revision does not match its content")
    return value


def validate_source_admission(
    admission: Mapping[str, Any],
    *,
    source_sha256: str,
    source_bytes: int,
) -> dict[str, Any]:
    """Bind an admission decision to the exact bytes inspected by an adapter."""

    required = {
        "source_manifest_revision",
        "source_ref_sha256",
        "source_snapshot_status",
        "source_sha256",
        "source_bytes",
    }
    missing = sorted(required.difference(admission))
    if missing:
        raise SourceAdmissionError(
            "source admission is missing required fields: " + ", ".join(missing)
        )
    digest = _optional_digest(admission["source_sha256"], "admission.source_sha256")
    expected_digest = _optional_digest(source_sha256, "source_sha256")
    if digest != expected_digest:
        raise SourceAdmissionError("source admission digest does not match inspected bytes")
    expected_bytes = _optional_bytes(admission["source_bytes"], "admission.source_bytes")
    if expected_bytes != source_bytes:
        raise SourceAdmissionError("source admission byte count does not match inspected bytes")
    return dict(admission)


class SourceManifestIndex:
    """Validated source-ref lookup used by bounded extractor admission."""

    def __init__(self, manifest: Mapping[str, Any]) -> None:
        if manifest.get("schema_version") != SOURCE_MANIFEST_SCHEMA:
            raise SourceAdmissionError("unsupported source manifest schema")
        if manifest.get("authority") != "stable_snapshot":
            raise SourceAdmissionError("source manifest authority must be stable_snapshot")
        self.revision = _verify_manifest_revision(manifest)
        raw_files = manifest.get("files")
        if not isinstance(raw_files, list):
            raise SourceAdmissionError("source manifest files must be a list")
        self._by_digest: dict[str, list[dict[str, Any]]] = {}
        self._by_identity: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
        for index, raw in enumerate(raw_files):
            if not isinstance(raw, dict):
                raise SourceAdmissionError(f"source manifest files[{index}] must be an object")
            source_sha256 = _optional_digest(raw.get("source_sha256"), f"files[{index}].source_sha256")
            source_ref_sha256 = _optional_digest(
                raw.get("source_ref_sha256"),
                f"files[{index}].source_ref_sha256",
            )
            if source_sha256 is None or source_ref_sha256 is None:
                raise SourceAdmissionError(f"source manifest files[{index}] lacks identity")
            provider = canonical_provider(_require_text(raw.get("provider"), f"files[{index}].provider"))
            root_label = canonical_root_label(_require_text(raw.get("root_label"), f"files[{index}].root_label"))
            source_class = canonical_source_class(
                _require_text(raw.get("source_class"), f"files[{index}].source_class")
            )
            relative_path_sha256 = _optional_digest(
                raw.get("relative_path_sha256"),
                f"files[{index}].relative_path_sha256",
            )
            if relative_path_sha256 is None:
                raise SourceAdmissionError(f"source manifest files[{index}] lacks path identity")
            without_ref = dict(raw)
            without_ref.pop("source_ref_sha256", None)
            if source_ref_sha256 != _digest_value(without_ref):
                raise SourceAdmissionError(f"source ref digest mismatch at files[{index}]")
            row = dict(raw)
            self._by_digest.setdefault(source_sha256, []).append(row)
            self._by_identity.setdefault(
                (provider, root_label, source_class, relative_path_sha256),
                [],
            ).append(row)

    @classmethod
    def from_path(cls, path: Path) -> "SourceManifestIndex":
        return cls(_load_json(Path(path)))

    def admit(
        self,
        *,
        source_sha256: str,
        provider: str,
        root_label: str,
        source_class: str,
    ) -> dict[str, Any]:
        digest = _optional_digest(source_sha256, "source_sha256")
        if digest is None:
            raise SourceAdmissionError("source_sha256 is required for admission")
        candidates = self._by_digest.get(digest, [])
        if not candidates:
            raise SourceAdmissionError("source digest is absent from the source manifest")
        expected_provider = canonical_provider(provider)
        expected_class = canonical_source_class(source_class)
        expected_root = canonical_root_label(root_label)
        matches = [
            row
            for row in candidates
            if row.get("provider") == expected_provider
            and row.get("root_label") == expected_root
            and row.get("source_class") == expected_class
        ]
        if len(matches) != 1:
            raise SourceAdmissionError(
                "source identity metadata is ambiguous or mismatched for the supplied digest"
            )
        row = matches[0]
        if row.get("snapshot_status") not in ADMISSIBLE_SNAPSHOT_STATUSES:
            raise SourceAdmissionError(
                f"source snapshot status is not admissible: {row.get('snapshot_status')}"
            )
        return {
            "source_manifest_revision": self.revision,
            "source_ref_sha256": row["source_ref_sha256"],
            "source_snapshot_status": row["snapshot_status"],
            "source_sha256": row["source_sha256"],
            "source_bytes": row["source_bytes"],
            "provider": row["provider"],
            "root_label": row["root_label"],
            "source_class": row["source_class"],
        }

    def admit_path(
        self,
        *,
        source_sha256: str,
        relative_path_sha256: str,
        provider: str,
        root_label: str,
        source_class: str,
    ) -> dict[str, Any]:
        """Admit a source using both bytes and its immutable relative identity."""

        digest = _optional_digest(source_sha256, "source_sha256")
        relative = _optional_digest(relative_path_sha256, "relative_path_sha256")
        if digest is None or relative is None:
            raise SourceAdmissionError("path admission requires source and path digests")
        identity = (
            canonical_provider(provider),
            canonical_root_label(root_label),
            canonical_source_class(source_class),
            relative,
        )
        matches = self._by_identity.get(identity, [])
        if len(matches) != 1:
            raise SourceAdmissionError(
                "source path identity is absent or ambiguous in the source manifest"
            )
        row = matches[0]
        if _optional_digest(row.get("source_sha256"), "source_sha256") != digest:
            raise SourceAdmissionError(
                "source path identity digest does not match inspected bytes"
            )
        if row.get("snapshot_status") not in ADMISSIBLE_SNAPSHOT_STATUSES:
            raise SourceAdmissionError(
                f"source snapshot status is not admissible: {row.get('snapshot_status')}"
            )
        return {
            "source_manifest_revision": self.revision,
            "source_ref_sha256": row["source_ref_sha256"],
            "source_snapshot_status": row["snapshot_status"],
            "source_sha256": row["source_sha256"],
            "source_bytes": row["source_bytes"],
            "provider": row["provider"],
            "root_label": row["root_label"],
            "source_class": row["source_class"],
            "relative_path_sha256": row["relative_path_sha256"],
        }


def write_manifest(path: Path, manifest: dict[str, Any], *, overwrite: bool = False) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            json.dump(manifest, destination, ensure_ascii=False, indent=2)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = build_source_manifest(args.inventory, args.snapshot)
        write_manifest(args.output, manifest, overwrite=args.overwrite)
    except (FileNotFoundError, OSError, SourceManifestError, FileExistsError) as exc:
        print(f"Source manifest failed: {exc}")
        return 2
    counts = manifest["counts"]
    print(
        f"Source manifest wrote {args.output}: {counts['joined_files']} joined, "
        f"{counts['changed_since_inventory']} changed since inventory, "
        f"{counts['unaccounted_files']} unaccounted"
    )
    return 0 if counts["unaccounted_files"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADMISSIBLE_SNAPSHOT_STATUSES",
    "PROVIDER_ALIASES",
    "ROOT_LABEL_ALIASES",
    "SOURCE_MANIFEST_SCHEMA",
    "SOURCE_MANIFEST_VERSION",
    "SOURCE_CLASS_ALIASES",
    "SourceAdmissionError",
    "SourceManifestError",
    "SourceManifestIndex",
    "SourceRef",
    "build_source_manifest",
    "canonical_root_label",
    "canonical_provider",
    "canonical_source_class",
    "validate_source_admission",
    "write_manifest",
]
