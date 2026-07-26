"""Prune inactive tenant-isolated Honcho workspaces."""

from __future__ import annotations

import os
import time
from pathlib import Path

from plugins.memory.honcho.client import (
    HonchoClientConfig,
    delete_tenant_workspace,
)
from plugins.memory.honcho.tenant_policy import TenantPolicyStore


def run_retention(
    config: HonchoClientConfig,
    *,
    now_ms: int,
) -> list[str]:
    if config.tenant_policy_enabled is not True:
        raise ValueError("tenant policy is not enabled")
    secret_value = os.environ.get(config.tenant_secret_env, "")
    if len(secret_value.encode("utf-8")) < 32:
        raise ValueError(f"{config.tenant_secret_env} must contain at least 32 bytes")
    if not config.tenant_policy_version or not config.tenant_policy_database:
        raise ValueError(
            "tenantPolicyVersion and tenantPolicyDatabase are required"
        )
    with TenantPolicyStore(
        Path(config.tenant_policy_database),
        secret=secret_value.encode("utf-8"),
        policy_version=config.tenant_policy_version,
    ) as store:
        return store.prune_inactive(
            now_ms=now_ms,
            delete_workspace=lambda workspace_id: delete_tenant_workspace(
                config,
                workspace_id,
            ),
        )


def main() -> None:
    config = HonchoClientConfig.from_global_config()
    deleted = run_retention(config, now_ms=int(time.time() * 1000))
    print(f"pruned tenant workspaces: {len(deleted)}")


if __name__ == "__main__":
    main()
