"""Durable model-artifact store backed by Postgres (Supabase).

Model binaries (joblib bundles) are persisted as ``bytea`` in ``public.model_artifacts``,
keyed by the same ``version_key`` used in ``public.model_versions``. This decouples the
artifact from the ephemeral training host: any environment (CI runner, local, dashboard)
can hydrate the active model on demand instead of relying on a local file that may not
exist, which previously forced a silent Poisson fallback.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from sqlalchemy import text

from alttez.api.db import get_engine


LOGGER = logging.getLogger(__name__)


def upload_artifact(version_key: str, local_path: str | Path, *, database_url: str | None = None) -> bool:
    """Persists a local artifact file into the durable store, keyed by ``version_key``.

    Idempotent: re-uploading the same ``version_key`` overwrites the stored bytes.
    Returns ``True`` on success.
    """

    path = Path(local_path)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    engine = get_engine(database_url)
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                insert into public.model_artifacts
                    (version_key, artifact_name, artifact_bytes, byte_size, sha256)
                values (:version_key, :artifact_name, :artifact_bytes, :byte_size, :sha256)
                on conflict (version_key) do update set
                    artifact_name  = excluded.artifact_name,
                    artifact_bytes = excluded.artifact_bytes,
                    byte_size      = excluded.byte_size,
                    sha256         = excluded.sha256,
                    created_at     = now()
                """
            ),
            {
                "version_key": version_key,
                "artifact_name": path.name,
                "artifact_bytes": data,
                "byte_size": len(data),
                "sha256": digest,
            },
        )
    LOGGER.info("artifact_uploaded version_key=%s bytes=%s", version_key, len(data))
    return True


def download_artifact(version_key: str, dest_path: str | Path, *, database_url: str | None = None) -> Path | None:
    """Hydrates the artifact for ``version_key`` to ``dest_path``.

    Returns the destination path on success, or ``None`` if no artifact is stored for the
    version (so callers can keep their existing fallback path).
    """

    engine = get_engine(database_url)
    with engine.connect() as connection:
        row = connection.execute(
            text("select artifact_bytes from public.model_artifacts where version_key = :version_key"),
            {"version_key": version_key},
        ).fetchone()

    if row is None or row[0] is None:
        return None

    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(bytes(row[0]))
    LOGGER.info("artifact_downloaded version_key=%s dest=%s", version_key, dest)
    return dest
