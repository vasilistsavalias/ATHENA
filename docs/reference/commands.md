# Commands Reference

## Pipeline

Full run:

```bash
./scripts/pipeline/setup_and_run.sh --full
```

Smoke test:

```bash
./scripts/pipeline/setup_and_run.sh --smoke-test
```

Resume:

```bash
./scripts/pipeline/setup_and_run.sh --resume --stages S12 S13 S14 S15 S16 S17 S18
```

Fresh start:

```bash
./scripts/pipeline/clean_all_outputs.sh
```

## Repository tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Website

API:

```powershell
cd website/services/api
python -m pytest tests -q
```

Web:

```powershell
cd website/services/web
npm run lint
npm run test
npm run build
```

## GCP

Sanitized execution flow:

```bash
git fetch origin
git checkout main
git reset --hard origin/main
chmod +x scripts/pipeline/setup_and_run.sh
./scripts/pipeline/setup_and_run.sh --full
```

Europeana-resilient flow (continue on Europeana shortfall/failure):

```bash
cp config/pipeline/main_config.yaml config/pipeline/gcp_continue_on_europeana_shortfall.yaml
sed -i 's/strict_fail_policy: true/strict_fail_policy: false/' config/pipeline/gcp_continue_on_europeana_shortfall.yaml
sed -i 's/require_europeana_api_key: true/require_europeana_api_key: false/' config/pipeline/gcp_continue_on_europeana_shortfall.yaml
sed -i 's/require_enabled_source_nonzero: true/require_enabled_source_nonzero: false/' config/pipeline/gcp_continue_on_europeana_shortfall.yaml
sed -i 's/require_europeana_key_when_enabled: true/require_europeana_key_when_enabled: false/' config/pipeline/gcp_continue_on_europeana_shortfall.yaml
EUROPEANA_API_KEY='<your-key>' ./scripts/pipeline/setup_and_run.sh --full --config config/pipeline/gcp_continue_on_europeana_shortfall.yaml
```

Optional: keep copyright-free mode and force a fresh Europeana cursor each run:

```bash
sed -i 's/europeana_reset_state_on_run: false/europeana_reset_state_on_run: true/' config/pipeline/gcp_continue_on_europeana_shortfall.yaml
sed -i 's/europeana_backup_state_before_reset: true/europeana_backup_state_before_reset: true/' config/pipeline/gcp_continue_on_europeana_shortfall.yaml
# Keep rights-safe mode enabled
sed -i 's/europeana_copyright_free_only: false/europeana_copyright_free_only: true/' config/pipeline/gcp_continue_on_europeana_shortfall.yaml
```

Canonical sources:

- `scripts/pipeline/setup_and_run.sh`
- `run.py`
- `website/services/api/app/tools/import_pack.py`
