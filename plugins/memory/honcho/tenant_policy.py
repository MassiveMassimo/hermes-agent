"""Consent and tenant isolation policy for multi-user Honcho gateways."""

from __future__ import annotations

import hashlib
import hmac
import sqlite3
import threading
from pathlib import Path
from typing import Callable


RETENTION_MS = 30 * 24 * 60 * 60 * 1000
MAX_FACT_CHARS = 500
ALLOWED_FACT_CATEGORIES = frozenset(
    {
        "approved_cv_fact",
        "role_preference",
        "writing_preference",
        "editing_decision",
    }
)
ACKNOWLEDGEMENT = "boleh diingat"
NOTICE_TTL_MS = 24 * 60 * 60 * 1000


class ConsentRequired(PermissionError):
    """Raised when the current memory policy has not been accepted."""


def tenant_workspace_id(sender_id: str, secret: bytes) -> str:
    """Derive a stable workspace ID without exposing the gateway sender ID."""
    sender = sender_id.strip()
    if not sender:
        raise ValueError("sender_id must not be empty")
    if len(secret) < 32:
        raise ValueError("tenant key secret must be at least 32 bytes")
    digest = hmac.new(secret, sender.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"wa_{digest[:32]}"


def validate_fact(category: str, content: str) -> tuple[str, str]:
    """Accept one concise fact from an explicitly approved product category."""
    normalized_category = category.strip()
    normalized_content = content.strip()
    if normalized_category not in ALLOWED_FACT_CATEGORIES:
        raise ValueError("memory fact category is not allowed")
    if not normalized_content or len(normalized_content) > MAX_FACT_CHARS:
        raise ValueError("memory fact must contain 1 to 500 characters")
    if "\n" in normalized_content or "\r" in normalized_content:
        raise ValueError("memory fact must be a single line")
    return normalized_category, normalized_content


class TenantPolicyStore:
    """Persist only opaque tenant consent and last-activity timestamps."""

    def __init__(
        self,
        database_path: Path,
        *,
        secret: bytes,
        policy_version: str,
        retention_ms: int = RETENTION_MS,
    ) -> None:
        if not policy_version.strip():
            raise ValueError("policy_version must not be empty")
        if retention_ms <= 0:
            raise ValueError("retention_ms must be positive")
        self._secret = secret
        self._policy_version = policy_version
        self._retention_ms = retention_ms
        database_path.parent.mkdir(parents=True, exist_ok=True)
        database_path.parent.chmod(0o700)
        self._database_path = database_path
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(database_path, check_same_thread=False)
        self.connection.execute("PRAGMA secure_delete=ON")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tenant_policy (
                tenant_key TEXT PRIMARY KEY,
                policy_version TEXT,
                consented_at_ms INTEGER CHECK (consented_at_ms >= 0),
                notice_pending_at_ms INTEGER CHECK (notice_pending_at_ms >= 0),
                last_activity_ms INTEGER NOT NULL CHECK (last_activity_ms >= 0)
            )
            """
        )
        self.connection.commit()
        self._secure_files()

    def __enter__(self) -> TenantPolicyStore:
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def _key(self, sender_id: str) -> str:
        return tenant_workspace_id(sender_id, self._secret)

    def grant_consent(self, sender_id: str, *, now_ms: int) -> str:
        key = self._key(sender_id)
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO tenant_policy (
                    tenant_key, policy_version, consented_at_ms,
                    notice_pending_at_ms, last_activity_ms
                ) VALUES (?, ?, ?, NULL, ?)
                ON CONFLICT(tenant_key) DO UPDATE SET
                    policy_version = excluded.policy_version,
                    consented_at_ms = excluded.consented_at_ms,
                    notice_pending_at_ms = NULL,
                    last_activity_ms = excluded.last_activity_ms
                """,
                (key, self._policy_version, now_ms, now_ms),
            )
            self.connection.commit()
            self._secure_files()
        return key

    def mark_notice(self, sender_id: str, *, now_ms: int) -> None:
        key = self._key(sender_id)
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO tenant_policy (
                    tenant_key, policy_version, consented_at_ms,
                    notice_pending_at_ms, last_activity_ms
                ) VALUES (?, NULL, NULL, ?, ?)
                ON CONFLICT(tenant_key) DO UPDATE SET
                    notice_pending_at_ms = excluded.notice_pending_at_ms
                """,
                (key, now_ms, now_ms),
            )
            self.connection.commit()
            self._secure_files()

    def accept_acknowledgement(
        self,
        sender_id: str,
        message: str,
        *,
        now_ms: int,
    ) -> bool:
        if message.strip().casefold() != ACKNOWLEDGEMENT:
            return False
        key = self._key(sender_id)
        with self._lock:
            row = self.connection.execute(
                """
                SELECT notice_pending_at_ms
                FROM tenant_policy
                WHERE tenant_key = ?
                """,
                (key,),
            ).fetchone()
            if (
                row is None
                or row[0] is None
                or now_ms - row[0] > NOTICE_TTL_MS
            ):
                return False
            self.connection.execute(
                """
                UPDATE tenant_policy
                SET policy_version = ?, consented_at_ms = ?,
                    notice_pending_at_ms = NULL, last_activity_ms = ?
                WHERE tenant_key = ?
                """,
                (self._policy_version, now_ms, now_ms, key),
            )
            self.connection.commit()
            self._secure_files()
            return True

    def require_access(self, sender_id: str, *, now_ms: int) -> str:
        key = self._key(sender_id)
        with self._lock:
            row = self.connection.execute(
                """
                SELECT policy_version, consented_at_ms
                FROM tenant_policy
                WHERE tenant_key = ?
                """,
                (key,),
            ).fetchone()
            if (
                row is None
                or row[0] != self._policy_version
                or row[1] is None
            ):
                raise ConsentRequired("current memory policy consent is required")
            self.connection.execute(
                "UPDATE tenant_policy SET last_activity_ms = ? WHERE tenant_key = ?",
                (now_ms, key),
            )
            self.connection.commit()
            self._secure_files()
        return key

    def forget(
        self,
        sender_id: str,
        *,
        delete_workspace: Callable[[str], None],
    ) -> None:
        key = self._key(sender_id)
        with self._lock:
            row = self.connection.execute(
                "SELECT tenant_key FROM tenant_policy WHERE tenant_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return
        delete_workspace(key)
        with self._lock:
            self.connection.execute(
                "DELETE FROM tenant_policy WHERE tenant_key = ?",
                (key,),
            )
            self.connection.commit()
            self._secure_files()

    def prune_inactive(
        self,
        *,
        now_ms: int,
        delete_workspace: Callable[[str], None],
    ) -> list[str]:
        cutoff = now_ms - self._retention_ms
        with self._lock:
            keys = [
                row[0]
                for row in self.connection.execute(
                    """
                    SELECT tenant_key
                    FROM tenant_policy
                    WHERE consented_at_ms IS NOT NULL
                      AND last_activity_ms < ?
                    ORDER BY tenant_key
                    """,
                    (cutoff,),
                )
            ]
        deleted: list[str] = []
        for key in keys:
            delete_workspace(key)
            with self._lock:
                self.connection.execute(
                    "DELETE FROM tenant_policy WHERE tenant_key = ?",
                    (key,),
                )
                self.connection.commit()
                self._secure_files()
            deleted.append(key)
        return deleted

    def _secure_files(self) -> None:
        for path in (
            self._database_path,
            Path(str(self._database_path) + "-wal"),
            Path(str(self._database_path) + "-shm"),
        ):
            if path.exists():
                path.chmod(0o600)
