# Public release checklist

Before changing GitHub repository visibility:

1. Review current changes (`git status --short`, `git diff`) and verify that the patch was applied to the intended repository version.
2. Review every branch and tag and scan the entire Git history for secrets, credentials, proprietary strategy files and licensed data. An archive of the current tree cannot establish that history is safe. If anything sensitive was committed, rotate credentials and remove the sensitive history before making the repo public. Do not publish secret-scan output.
3. Confirm market-data redistribution rights and inspect any aggregate reports before committing them. The repository should not contain raw vendor data, credentials, or private strategies.
4. Run `uv sync --locked`, `uv run --locked ruff check .`, `uv run --locked ruff format --check .`, `uv run --locked mypy src/quant_research`, and `uv run --locked pytest`. Verify GitHub Actions on the private repository first. The offline demo should produce the table shown in the README.
5. If permitted, add aggregate empirical scores from existing saved reports to `results/README.md` with dates, samples, costs and uncertainty. Do not use the protected holdout for this purpose.
6. Choose a license deliberately. Public visibility does not by itself grant an open-source license; do not add a permissive license unless that is the intended distribution policy.
7. Change repository visibility only after the checks pass. Confirm the README and links work when signed out.

The analytical demonstration does not need a data subscription. The full historical research workflows require the original local inputs and may require authenticated data access.
