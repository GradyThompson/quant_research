"""Create S3-compatible Massive Flat Files clients without credential leakage.

Massive's REST API token and its S3 secret key are different credentials.
Do not silently substitute MASSIVE_API_KEY for a missing S3 secret.
"""

from __future__ import annotations

import os
from typing import Any

import boto3  # type: ignore[import-untyped]
from botocore.config import Config  # type: ignore[import-untyped]
from dotenv import load_dotenv

from quant_research.paths import PROJECT_ROOT

MASSIVE_S3_ENDPOINT = "https://files.massive.com"


def create_massive_s3_client(*, aws_profile: str | None = None) -> Any:
    """Load project .env and resolve one complete S3 credential source.

    Priority: explicit Massive S3 key pair, standard AWS environment key pair,
    then Boto3's regular credential provider chain (shared file / role / profile).
    An explicit --aws-profile cannot be mixed with environment credentials.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    access_id = os.getenv("MASSIVE_S3_ACCESS_KEY_ID") or os.getenv("ACCESS_KEY_ID")
    secret_key = os.getenv("MASSIVE_S3_SECRET_ACCESS_KEY")
    aws_access_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    if aws_profile is not None:
        if any((access_id, secret_key, aws_access_id, aws_secret_key)):
            raise ValueError(
                "--aws-profile cannot be combined with S3 credentials in the environment/.env"
            )
        session = boto3.Session(profile_name=aws_profile)
    elif access_id or secret_key:
        if not access_id or not secret_key:
            raise ValueError(
                "Incomplete Massive S3 credentials: set ACCESS_KEY_ID (or "
                "MASSIVE_S3_ACCESS_KEY_ID) AND MASSIVE_S3_SECRET_ACCESS_KEY in .env. "
                "MASSIVE_API_KEY is a REST token, not the S3 secret key."
            )
        session = boto3.Session(
            aws_access_key_id=access_id,
            aws_secret_access_key=secret_key,
        )
    elif aws_access_id or aws_secret_key:
        if not aws_access_id or not aws_secret_key:
            raise ValueError(
                "Incomplete AWS S3 credentials: both AWS_ACCESS_KEY_ID and "
                "AWS_SECRET_ACCESS_KEY are required."
            )
        session = boto3.Session()
    else:
        session = boto3.Session()

    if session.get_credentials() is None:
        raise ValueError(
            "No Massive Flat Files S3 credentials found. Set MASSIVE_S3_ACCESS_KEY_ID "
            "and MASSIVE_S3_SECRET_ACCESS_KEY in the project .env, or configure an "
            "AWS shared-credentials profile. MASSIVE_API_KEY alone is not sufficient."
        )

    return session.client(
        "s3",
        endpoint_url=MASSIVE_S3_ENDPOINT,
        region_name="us-east-1",
        config=Config(signature_version="s3v4"),
    )
