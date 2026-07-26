"""Consent and tenant isolation policy for multi-user Honcho gateways."""

from __future__ import annotations

import hashlib
import hmac
import sqlite3
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
        self.connection = sqlite3.connect(database_path)
        self.connection.execute("PRAGMA secure_delete=ON")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tenant_policy (
                tenant_key TEXT PRIMARY KEY,
                policy_version TEXT NOT NULL,
                consented_at_ms INTEGER NOT NULL CHECK (consented_at_ms >= 0),
                last_activity_ms INTEGER NOT NULL CHECK (last_activity_ms >= 0)
            )
            """
        )
        self.connection.commit()
        database_path.chmod(0o600)

    def __enter__(self) -> TenantPolicyStore:
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def _key(self, sender_id: str) -> str:
        return tenant_workspace_id(sender_id, self._secret)

    def grant_consent(self, sender_id: str, *, now_ms: int) -> str:
        key = self._key(sender_id)
        self.connection.execute(
            """
            INSERT INTO tenant_policy (
                tenant_key, policy_version, consented_at_ms, last_activity_ms
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(tenant_key) DO UPDATE SET
                policy_version = excluded.policy_version,
                consented_at_ms = excluded.consented_at_ms,
                last_activity_ms = excluded.last_activity_ms
            """,
            (key, self._policy_version, now_ms, now_ms),
        )
        self.connection.commit()
        return key

    def require_access(self, sender_id: str, *, now_ms: int) -> str:
        key = self._key(sender_id)
        row = self.connection.execute(
            "SELECT policy_version FROM tenant_policy WHERE tenant_key = ?",
            (key,),
        ).fetchone()
        if row is None or row[0] != self._policy_version:
            raise ConsentRequired("current memory policy consent is required")
        self.connection.execute(
            "UPDATE tenant_policy SET last_activity_ms = ? WHERE tenant_key = ?",
            (now_ms, key),
        )
        self.connection.commit()
        return key

    def forget(
        self,
        sender_id: str,
        *,
        delete_workspace: Callable[[str], None],
    ) -> None:
        key = self._key(sender_id)
        row = self.connection.execute(
            "SELECT tenant_key FROM tenant_policy WHERE tenant_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return
        delete_workspace(key)
        self.connection.execute(
            "DELETE FROM tenant_policy WHERE tenant_key = ?",
            (key,),
        )
        self.connection.commit()

    def prune_inactive(
        self,
        *,
        now_ms: int,
        delete_workspace: Callable[[str], None],
    ) -> list[str]:
        cutoff = now_ms - self._retention_ms
        keys = [
            row[0]
            for row in self.connection.execute(
                """
                SELECT tenant_key
                FROM tenant_policy
                WHERE last_activity_ms < ?
                ORDER BY tenant_key
                """,
                (cutoff,),
            )
        ]
        deleted: list[str] = []
        for key in keys:
            delete_workspace(key)
            self.connection.execute(
                "DELETE FROM tenant_policy WHERE tenant_key = ?",
                (key,),
            )
            self.connection.commit()
            deleted.append(key)
        return deleted
