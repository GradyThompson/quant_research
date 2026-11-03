"""Offline Massive S3 credential resolution tests (no network or secrets printed)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from quant_research.data.ingestion import massive_s3


class StubSession:
    instances: list[StubSession] = []

    def __init__(self, **kwargs: object) -> None:
        self.settings = kwargs
        self.client_arguments: dict[str, object] = {}
        self.__class__.instances.append(self)

    def get_credentials(self) -> object | None:
        if self.settings or (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")):
            return object()
        return None

    def client(self, name: str, **kwargs: object) -> dict[str, object]:
        self.client_arguments = {"name": name, **kwargs}
        return self.client_arguments


@pytest.fixture(autouse=True)
def clean_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in (
        "MASSIVE_API_KEY",
        "ACCESS_KEY_ID",
        "MASSIVE_S3_ACCESS_KEY_ID",
        "MASSIVE_S3_SECRET_ACCESS_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_PROFILE",
    ):
        monkeypatch.delenv(name, raising=False)
    StubSession.instances.clear()
    monkeypatch.setattr(massive_s3.boto3, "Session", StubSession)
    monkeypatch.setattr(massive_s3, "PROJECT_ROOT", tmp_path)


def test_explicit_massive_s3_pair_loads_from_project_env(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "MASSIVE_API_KEY=rest-not-s3\n"
        "MASSIVE_S3_ACCESS_KEY_ID=test-access\n"
        "MASSIVE_S3_SECRET_ACCESS_KEY=test-secret\n",
        encoding="utf-8",
    )
    client = massive_s3.create_massive_s3_client()
    assert StubSession.instances[0].settings == {
        "aws_access_key_id": "test-access",
        "aws_secret_access_key": "test-secret",
    }
    assert client["name"] == "s3"
    assert client["endpoint_url"] == "https://files.massive.com"
    assert client["config"].signature_version == "s3v4"  # type: ignore[union-attr]


def test_existing_access_key_alias_is_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCESS_KEY_ID", "old-named-access")
    monkeypatch.setenv("MASSIVE_S3_SECRET_ACCESS_KEY", "flat-file-secret")
    massive_s3.create_massive_s3_client()
    assert StubSession.instances[0].settings["aws_access_key_id"] == "old-named-access"


def test_rest_token_is_not_silently_used_as_s3_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCESS_KEY_ID", "s3-access")
    monkeypatch.setenv("MASSIVE_API_KEY", "rest-token")
    with pytest.raises(ValueError, match="MASSIVE_API_KEY is a REST token"):
        massive_s3.create_massive_s3_client()
    assert not StubSession.instances


def test_missing_access_key_reports_both_required_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MASSIVE_S3_SECRET_ACCESS_KEY", "s3-secret")
    with pytest.raises(ValueError, match="ACCESS_KEY_ID"):
        massive_s3.create_massive_s3_client()


def test_standard_aws_environment_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "aws-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    massive_s3.create_massive_s3_client()
    assert StubSession.instances[0].settings == {}


def test_partial_standard_aws_pair_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "aws-access")
    with pytest.raises(ValueError, match="both AWS_ACCESS_KEY_ID"):
        massive_s3.create_massive_s3_client()


def test_profile_permitted_with_rest_key_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MASSIVE_API_KEY", "rest-token")
    massive_s3.create_massive_s3_client(aws_profile="massive-profile")
    assert StubSession.instances[0].settings == {"profile_name": "massive-profile"}


def test_profile_and_s3_credentials_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCESS_KEY_ID", "access")
    with pytest.raises(ValueError, match="--aws-profile cannot be combined"):
        massive_s3.create_massive_s3_client(aws_profile="profile")


def test_missing_all_credentials_reports_s3_instructions() -> None:
    with pytest.raises(ValueError, match="MASSIVE_S3_SECRET_ACCESS_KEY"):
        massive_s3.create_massive_s3_client()


def test_process_environment_credential_overrides_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text(
        "MASSIVE_S3_ACCESS_KEY_ID=file-access\nMASSIVE_S3_SECRET_ACCESS_KEY=file-secret\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MASSIVE_S3_ACCESS_KEY_ID", "environment-access")
    massive_s3.create_massive_s3_client()
    assert StubSession.instances[0].settings["aws_access_key_id"] == "environment-access"
