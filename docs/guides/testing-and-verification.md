# Testing and Verification

ATHENA uses different authoritative environments for different claims.

## Authoritative environments

- Full pipeline validation: Linux/GCP
- Full repository test suite: repository `.venv`
- Windows local: valid for unit/stage tests and website development

## Main commands

Full repo tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Website API:

```powershell
cd website/services/api
python -m pytest tests -q
```

Website web:

```powershell
cd website/services/web
npm run lint
npm run test
npm run build
```

## What green tests do not prove

- a green Windows unit suite does not prove Linux GPU correctness
- a green frontend build does not prove campaign-quality data
- a pilot export does not prove thesis-ready expert evidence

## Recommended verification sequence

1. run targeted unit/stage tests for changed code
2. run service-local tests for website changes
3. inspect artifact contracts and logs after pipeline runs
4. validate expert-pack import and admin export before expert collection

See also:

- [`../reference/observability.md`](../reference/observability.md)
- [`../reference/troubleshooting.md`](../reference/troubleshooting.md)
