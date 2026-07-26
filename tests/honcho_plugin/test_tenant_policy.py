from __future__ import annotations

import tempfile
from pathlib import Path
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from plugins.memory.honcho.tenant_policy import (
    ALLOWED_FACT_CATEGORIES,
    ConsentRequired,
    TenantPolicyStore,
    tenant_workspace_id,
    validate_fact,
)
from plugins.memory.honcho.client import (
    HonchoClientConfig,
    build_tenant_honcho_client,
)
from plugins.memory.honcho.session import HonchoSession, HonchoSessionManager
from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.tenant_retention import run_retention


SECRET = b"synthetic-test-secret-at-least-32-bytes"
POLICY_VERSION = "cv-memory-v1"
DAY_MS = 24 * 60 * 60 * 1000


def test_workspace_ids_are_stable_opaque_and_sender_specific():
    sender_a = "whatsapp-synthetic-user-a"
    sender_b = "whatsapp-synthetic-user-b"

    workspace_a = tenant_workspace_id(sender_a, SECRET)

    assert workspace_a == tenant_workspace_id(sender_a, SECRET)
    assert workspace_a != tenant_workspace_id(sender_b, SECRET)
    assert workspace_a.startswith("wa_")
    assert sender_a not in workspace_a


def test_access_requires_current_consent_without_creating_state(tmp_path):
    with TenantPolicyStore(
        tmp_path / "policy.sqlite3",
        secret=SECRET,
        policy_version=POLICY_VERSION,
    ) as store:
        with pytest.raises(ConsentRequired):
            store.require_access("sender-a", now_ms=100)

        count = store.connection.execute(
            "SELECT COUNT(*) FROM tenant_policy"
        ).fetchone()[0]
        assert count == 0


def test_consent_is_sender_scoped_and_survives_restart(tmp_path):
    database_path = tmp_path / "policy.sqlite3"
    with TenantPolicyStore(
        database_path,
        secret=SECRET,
        policy_version=POLICY_VERSION,
    ) as store:
        workspace_a = store.grant_consent("sender-a", now_ms=100)
        with pytest.raises(ConsentRequired):
            store.require_access("sender-b", now_ms=110)

    with TenantPolicyStore(
        database_path,
        secret=SECRET,
        policy_version=POLICY_VERSION,
    ) as store:
        assert store.require_access("sender-a", now_ms=120) == workspace_a


def test_database_never_persists_raw_sender_id(tmp_path):
    raw_sender = "sender-secret-value"
    database_path = tmp_path / "policy.sqlite3"
    with TenantPolicyStore(
        database_path,
        secret=SECRET,
        policy_version=POLICY_VERSION,
    ) as store:
        store.grant_consent(raw_sender, now_ms=100)

    assert raw_sender.encode() not in database_path.read_bytes()


def test_forget_deletes_workspace_before_local_state(tmp_path):
    with TenantPolicyStore(
        tmp_path / "policy.sqlite3",
        secret=SECRET,
        policy_version=POLICY_VERSION,
    ) as store:
        workspace = store.grant_consent("sender-a", now_ms=100)
        deleted = []

        store.forget("sender-a", delete_workspace=deleted.append)

        assert deleted == [workspace]
        with pytest.raises(ConsentRequired):
            store.require_access("sender-a", now_ms=200)


def test_failed_remote_deletion_preserves_state_for_retry(tmp_path):
    with TenantPolicyStore(
        tmp_path / "policy.sqlite3",
        secret=SECRET,
        policy_version=POLICY_VERSION,
    ) as store:
        workspace = store.grant_consent("sender-a", now_ms=100)

        def fail_delete(candidate: str) -> None:
            assert candidate == workspace
            raise RuntimeError("synthetic delete failure")

        with pytest.raises(RuntimeError, match="synthetic delete failure"):
            store.forget("sender-a", delete_workspace=fail_delete)

        assert store.require_access("sender-a", now_ms=200) == workspace


def test_retention_deletes_only_inactive_tenants(tmp_path):
    with TenantPolicyStore(
        tmp_path / "policy.sqlite3",
        secret=SECRET,
        policy_version=POLICY_VERSION,
    ) as store:
        stale = store.grant_consent("sender-stale", now_ms=0)
        fresh = store.grant_consent("sender-fresh", now_ms=20 * DAY_MS)
        deleted = []

        result = store.prune_inactive(
            now_ms=31 * DAY_MS,
            delete_workspace=deleted.append,
        )

        assert result == [stale]
        assert deleted == [stale]
        with pytest.raises(ConsentRequired):
            store.require_access("sender-stale", now_ms=31 * DAY_MS)
        assert (
            store.require_access("sender-fresh", now_ms=31 * DAY_MS)
            == fresh
        )


@pytest.mark.parametrize("category", sorted(ALLOWED_FACT_CATEGORIES))
def test_allowed_fact_categories_accept_one_bounded_fact(category):
    assert validate_fact(category, "Prefers a concise professional summary.") == (
        category,
        "Prefers a concise professional summary.",
    )


@pytest.mark.parametrize(
    ("category", "content"),
    [
        ("raw_cv", "Synthetic CV body"),
        ("role_preference", "line one\nline two"),
        ("role_preference", "x" * 501),
        ("role_preference", "   "),
    ],
)
def test_fact_validation_rejects_unapproved_or_document_like_content(
    category,
    content,
):
    with pytest.raises(ValueError):
        validate_fact(category, content)


def test_tenant_policy_config_is_disabled_by_default():
    config = HonchoClientConfig()

    assert config.tenant_policy_enabled is False
    assert config.tenant_policy_version == ""
    assert config.tenant_policy_database == ""
    assert config.tenant_secret_env == "HONCHO_TENANT_SECRET"


def test_tenant_policy_config_parses_from_profile_host(tmp_path):
    config_path = tmp_path / "honcho.json"
    config_path.write_text(
        json.dumps(
            {
                "apiKey": "synthetic",
                "hosts": {
                    "hermes": {
                        "tenantPolicyEnabled": True,
                        "tenantPolicyVersion": POLICY_VERSION,
                        "tenantPolicyDatabase": "/private/policy.sqlite3",
                        "tenantSecretEnv": "LOKERKIT_MEMORY_TENANT_SECRET",
                    }
                },
            }
        )
    )

    config = HonchoClientConfig.from_global_config(config_path=config_path)

    assert config.tenant_policy_enabled is True
    assert config.tenant_policy_version == POLICY_VERSION
    assert config.tenant_policy_database == "/private/policy.sqlite3"
    assert config.tenant_secret_env == "LOKERKIT_MEMORY_TENANT_SECRET"


def test_session_manager_preserves_an_injected_tenant_client():
    client = MagicMock()
    manager = HonchoSessionManager(
        honcho=client,
        config=HonchoClientConfig(write_frequency="turn"),
    )

    assert manager.honcho is client


def test_tenant_policy_rejects_model_supplied_peer_ids():
    manager = HonchoSessionManager(
        honcho=MagicMock(),
        config=HonchoClientConfig(
            tenant_policy_enabled=True,
            write_frequency="turn",
        ),
    )
    session = HonchoSession(
        key="whatsapp_cloud:synthetic",
        user_peer_id="user",
        assistant_peer_id="assistant",
        honcho_session_id="synthetic",
    )

    assert manager._resolve_peer_id(session, "user") == "user"
    assert manager._resolve_peer_id(session, "ai") == "assistant"
    with pytest.raises(ValueError, match="current user"):
        manager._resolve_peer_id(session, "another-tenant")


def _tenant_config(database_path: Path) -> HonchoClientConfig:
    return HonchoClientConfig(
        api_key="synthetic",
        base_url="http://127.0.0.1:8000",
        enabled=True,
        recall_mode="tools",
        init_on_session_start=False,
        save_messages=False,
        tenant_policy_enabled=True,
        tenant_policy_version=POLICY_VERSION,
        tenant_policy_database=str(database_path),
        tenant_secret_env="LOKERKIT_MEMORY_TENANT_SECRET",
    )


def test_unconsented_gateway_user_gets_only_privacy_tool(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "LOKERKIT_MEMORY_TENANT_SECRET",
        SECRET.decode(),
    )
    provider = HonchoMemoryProvider()
    config = _tenant_config(tmp_path / "policy.sqlite3")

    with patch(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        return_value=config,
    ):
        provider.initialize(
            session_id="synthetic-session",
            platform="whatsapp_cloud",
            user_id="sender-a",
        )

    assert [tool["name"] for tool in provider.get_tool_schemas()] == [
        "honcho_privacy"
    ]
    assert provider._manager is None


def test_privacy_accept_enables_bounded_memory_tools(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "LOKERKIT_MEMORY_TENANT_SECRET",
        SECRET.decode(),
    )
    provider = HonchoMemoryProvider()
    config = _tenant_config(tmp_path / "policy.sqlite3")

    with patch(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        return_value=config,
    ):
        provider.initialize(
            session_id="synthetic-session",
            platform="whatsapp_cloud",
            user_id="sender-a",
        )

    result = provider.handle_tool_call(
        "honcho_privacy",
        {"action": "accept"},
    )

    assert '"consented": true' in result
    assert [tool["name"] for tool in provider.get_tool_schemas()] == [
        "honcho_privacy",
        "honcho_profile",
        "honcho_conclude",
    ]


def test_tenant_client_is_constructed_for_the_opaque_workspace(monkeypatch):
    constructor = MagicMock(return_value=MagicMock())
    monkeypatch.setitem(sys.modules, "honcho", SimpleNamespace(Honcho=constructor))
    config = _tenant_config(Path("/private/policy.sqlite3"))

    build_tenant_honcho_client(config, "wa_opaque")

    assert constructor.call_args.kwargs["workspace_id"] == "wa_opaque"
    assert constructor.call_args.kwargs["base_url"] == "http://127.0.0.1:8000"
    assert constructor.call_args.kwargs["api_key"] == "synthetic"


def test_consented_session_uses_opaque_workspace_and_no_sender_peer(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "LOKERKIT_MEMORY_TENANT_SECRET",
        SECRET.decode(),
    )
    provider = HonchoMemoryProvider()
    config = _tenant_config(tmp_path / "policy.sqlite3")
    manager = MagicMock()
    manager.get_or_create.return_value = SimpleNamespace(messages=[])

    with patch(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        return_value=config,
    ):
        provider.initialize(
            session_id="synthetic-session",
            platform="whatsapp_cloud",
            user_id="sender-a",
        )
    provider.handle_tool_call("honcho_privacy", {"action": "accept"})

    with (
        patch(
            "plugins.memory.honcho.client.build_tenant_honcho_client",
            return_value=MagicMock(),
        ) as build_client,
        patch(
            "plugins.memory.honcho.session.HonchoSessionManager",
            return_value=manager,
        ) as manager_class,
        patch("hermes_constants.get_hermes_home", return_value=tmp_path),
    ):
        assert provider._ensure_session() is True

    workspace = tenant_workspace_id("sender-a", SECRET)
    assert build_client.call_args.args[1] == workspace
    manager_config = manager_class.call_args.kwargs["config"]
    assert manager_config.workspace_id == workspace
    assert manager_config.peer_name == "user"
    assert manager_config.ai_peer == "assistant"
    assert manager_class.call_args.kwargs["runtime_user_peer_name"] is None
    assert provider._session_key == "cv-memory"


def _ready_tenant_provider() -> tuple[HonchoMemoryProvider, MagicMock]:
    provider = HonchoMemoryProvider()
    provider._config = _tenant_config(Path("/private/policy.sqlite3"))
    provider._tenant_consented = True
    provider._session_initialized = True
    provider._session_key = "cv-memory"
    manager = MagicMock()
    provider._manager = manager
    return provider, manager


def test_tenant_conclusion_requires_an_allowed_fact_category():
    provider, manager = _ready_tenant_provider()

    missing = provider.handle_tool_call(
        "honcho_conclude",
        {"conclusion": "Prefers concise summaries."},
    )
    disallowed = provider.handle_tool_call(
        "honcho_conclude",
        {
            "category": "raw_cv",
            "conclusion": "Synthetic raw CV body",
        },
    )

    assert '"error"' in missing
    assert '"error"' in disallowed
    manager.create_conclusion.assert_not_called()


def test_tenant_conclusion_persists_one_validated_fact():
    provider, manager = _ready_tenant_provider()
    manager.create_conclusion.return_value = True

    result = provider.handle_tool_call(
        "honcho_conclude",
        {
            "category": "writing_preference",
            "conclusion": "Prefers concise summaries.",
        },
    )

    assert '"error"' not in result
    manager.create_conclusion.assert_called_once_with(
        "cv-memory",
        "[writing_preference] Prefers concise summaries.",
        peer="user",
    )


def test_tenant_profile_tool_is_read_only():
    provider, manager = _ready_tenant_provider()

    result = provider.handle_tool_call(
        "honcho_profile",
        {"card": ["Unreviewed arbitrary profile content"]},
    )

    assert '"error"' in result
    manager.set_peer_card.assert_not_called()


def test_privacy_forget_deletes_whole_workspace_and_revokes_consent(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "LOKERKIT_MEMORY_TENANT_SECRET",
        SECRET.decode(),
    )
    provider = HonchoMemoryProvider()
    config = _tenant_config(tmp_path / "policy.sqlite3")
    with patch(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        return_value=config,
    ):
        provider.initialize(
            session_id="synthetic-session",
            platform="whatsapp_cloud",
            user_id="sender-a",
        )
    provider.handle_tool_call("honcho_privacy", {"action": "accept"})

    with patch(
        "plugins.memory.honcho.client.delete_tenant_workspace"
    ) as delete_workspace:
        result = provider.handle_tool_call(
            "honcho_privacy",
            {"action": "forget"},
        )

    workspace = tenant_workspace_id("sender-a", SECRET)
    delete_workspace.assert_called_once_with(config, workspace)
    assert '"deleted": true' in result
    assert provider._tenant_consented is False
    assert provider._manager is None
    assert [tool["name"] for tool in provider.get_tool_schemas()] == [
        "honcho_privacy"
    ]


def test_failed_workspace_deletion_does_not_revoke_consent(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "LOKERKIT_MEMORY_TENANT_SECRET",
        SECRET.decode(),
    )
    provider = HonchoMemoryProvider()
    config = _tenant_config(tmp_path / "policy.sqlite3")
    with patch(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        return_value=config,
    ):
        provider.initialize(
            session_id="synthetic-session",
            platform="whatsapp_cloud",
            user_id="sender-a",
        )
    provider.handle_tool_call("honcho_privacy", {"action": "accept"})

    with patch(
        "plugins.memory.honcho.client.delete_tenant_workspace",
        side_effect=RuntimeError("synthetic deletion failure"),
    ):
        result = provider.handle_tool_call(
            "honcho_privacy",
            {"action": "forget"},
        )

    assert '"error"' in result
    assert provider._tenant_consented is True


def test_retention_runner_deletes_stale_workspace(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "LOKERKIT_MEMORY_TENANT_SECRET",
        SECRET.decode(),
    )
    config = _tenant_config(tmp_path / "policy.sqlite3")
    with TenantPolicyStore(
        tmp_path / "policy.sqlite3",
        secret=SECRET,
        policy_version=POLICY_VERSION,
    ) as store:
        stale = store.grant_consent("sender-stale", now_ms=0)

    with patch(
        "plugins.memory.honcho.tenant_retention.delete_tenant_workspace"
    ) as delete_workspace:
        deleted = run_retention(config, now_ms=31 * DAY_MS)

    assert deleted == [stale]
    delete_workspace.assert_called_once_with(config, stale)
