# Admin Operations

The admin surface is separate from the participant invite flow.

## Admin entrypoints

- UI login: `/admin/login`
- UI dashboard: `/admin`
- API fallback: `x-admin-secret` for scripted export/import access

## Main admin tasks

- inspect active campaign stats
- review participant progress
- review stage-end comments and item comments
- export CSV / JSON / quality report
- import a new expert pack

## Import policy

Import only final or approved pilot packs.

The backend blocks packs containing out-of-policy `>50%` `mask_coverage_bin` values.

## Operational checklist

Before inviting experts:

1. import the intended campaign pack
2. verify campaign counts and comments view in `/admin`
3. verify exports work
4. run one manual participant pass end to end

## Privacy and release note

The admin dashboard can expose optional personal data if participants chose to provide it. Treat exports as controlled research data, not repo content.

See also:

- [`api-contracts.md`](api-contracts.md)
- [`deployment.md`](deployment.md)
