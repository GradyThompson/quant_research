# P4.4.1: Massive Flat Files S3 credentials

The P4.4 downloader loaded `.env` but did not pass the project's custom
credential names to Boto3. A `botocore.exceptions.NoCredentialsError`
could occur before the first HEAD request.

## Setup

Obtain **Flat Files S3** credentials from the Massive dashboard. These are
not automatically the same as the `MASSIVE_API_KEY` used for the REST API.

Add to the repository-root `.env` (never commit it):

```dotenv
MASSIVE_API_KEY=<REST-token-if-used-by-other-research>
MASSIVE_S3_ACCESS_KEY_ID=<Massive-flat-file-S3-access-key>
MASSIVE_S3_SECRET_ACCESS_KEY=<Massive-flat-file-S3-secret-key>
```

Your existing name `ACCESS_KEY_ID` is also accepted as an alternative to
`MASSIVE_S3_ACCESS_KEY_ID`, **provided** you set
`MASSIVE_S3_SECRET_ACCESS_KEY` independently.

The downloader also accepts a standard `AWS_ACCESS_KEY_ID` +
`AWS_SECRET_ACCESS_KEY` pair and Boto3 AWS shared-credentials profiles
(`--aws-profile massive`). Do not mix `--aws-profile` with S3 environment
credentials. `MASSIVE_API_KEY` is never treated as an S3 secret automatically.

Credential values are not logged or placed into download manifests.

## Run

```bash
uv run python scripts/download_research_minute_flat_files.py \
    --start 2023-01-03 \
    --end 2024-12-31 \
    --max-sessions 25
```

The sibling `scripts/ingest_minute_flat_history.py` is also updated to use
this shared credential resolver. Neither change modifies the experimental
strategy or protected-date handling.

## Checks

```bash
uv run ruff format
uv run ruff check . --fix
uv run ruff format --check .
uv run mypy src/quant_research
```

```bash
uv run pytest
```

## Troubleshooting

- **Missing credential**: check the names in the repository-root `.env`.
- **403 Forbidden**: confirm you copied the dedicated Flat Files S3 secret,
  that your plan grants the requested data, and that the file/date exists.
- **SignatureDoesNotMatch**: verify the S3 access/secret pair (rather than
  mixing access key and REST token).
- **Older downloader**: `scripts/download_massive_daily_flat_file.py` and
  `scripts/import_massive_daily_window.py` retain their previous independent
  credential handling and are not changed by this focused patch.

Massive official documentation: https://massive.com/docs/flat-files/quickstart
