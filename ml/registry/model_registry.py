"""File-based model registry for tracking versions and artifacts.

Provides a lightweight alternative to the DB-based registry for
ML pipeline use where database access may not be available.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FileModelRegistry:
    """File-based model version registry.

    Stores version metadata as JSON files in the artifacts directory.
    Supports version listing, promotion, and rollback.
    """

    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.versions_dir = self.artifacts_dir / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    def register_version(
        self,
        version: str,
        model_type: str,
        artifact_path: str,
        metrics: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict:
        """Register a new model version.

        Args:
            version: Version identifier string.
            model_type: Model type (mf, ncf, lightgcn, cmf)
            artifact_path: Path to model artifacts.
            metrics: Evaluation metrics for this version.
            metadata: Additional metadata.

        Returns:
            Version record dict.
        """
        record = {
            "version": version,
            "model_type": model_type,
            "artifact_path": artifact_path,
            "metrics": metrics or {},
            "metadata": metadata or {},
            "is_active": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        version_dir = self.versions_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        with open(version_dir / "registry_entry.json", "w") as f:
            json.dump(record, f, indent=2)

        logger.info("Registered model version: %s (%s)", version, model_type)
        return record

    def get_version(self, version: str) -> dict | None:
        """Get a version record by version string."""
        entry_path = self.versions_dir / version / "registry_entry.json"
        if not entry_path.exists():
            return None
        with open(entry_path) as f:
            return json.load(f)

    def list_versions(
        self, model_type: str | None = None, limit: int = 20
    ) -> list[dict]:
        """List registered versions, newest first."""
        versions = []
        for version_dir in sorted(self.versions_dir.iterdir(), reverse=True):
            entry_path = version_dir / "registry_entry.json"
            if entry_path.exists():
                with open(entry_path) as f:
                    record = json.load(f)
                if model_type and record.get("model_type") != model_type:
                    continue
                versions.append(record)
                if len(versions) >= limit:
                    break
        return versions

    def get_active_version(self, model_type: str | None = None) -> dict | None:
        """Get the currently active version."""
        active_file = self.artifacts_dir / "active" / "version.txt"
        if not active_file.exists():
            return None
        version = active_file.read_text().strip()
        record = self.get_version(version)
        if record and model_type and record.get("model_type") != model_type:
            return None
        return record

    def promote(self, version: str) -> bool:
        """Promote a version to active status.

        Args:
            version: Version to promote.

        Returns:
            True if successful.
        """
        record = self.get_version(version)
        if not record:
            logger.error("Version %s not found", version)
            return False

        # Mark as active
        record["is_active"] = True
        entry_path = self.versions_dir / version / "registry_entry.json"
        with open(entry_path, "w") as f:
            json.dump(record, f, indent=2)

        # Update active pointer
        active_dir = self.artifacts_dir / "active"
        active_dir.mkdir(parents=True, exist_ok=True)
        with open(active_dir / "version.txt", "w") as f:
            f.write(version)

        logger.info("Promoted version %s to active", version)
        return True

    def rollback(self) -> str | None:
        """Rollback to the previous active version.

        Returns:
            The version rolled back to, or None if no previous version.
        """
        versions = self.list_versions()
        active_versions = [v for v in versions if v.get("is_active")]

        if len(versions) < 2:
            logger.warning("Not enough versions for rollback")
            return None

        # Find the second most recent version
        prev_version = versions[1]["version"]
        self.promote(prev_version)
        logger.info("Rolled back to version %s", prev_version)
        return prev_version
