# Security

## Public/private boundary

Public:

- code
- configs
- tests
- sanitized docs
- public fixtures

Local-only or private:

- `.env`
- deployment secrets
- runtime outputs
- live participant exports
- thesis-writing workspace
- `conductor/`

## Website-specific controls

- invite-code participant access
- cookie-backed participant sessions
- dedicated admin login
- admin secret for scripted export/import access

## Privacy note

The website can store optional personal data if experts choose to provide it:

- `name`
- `institution`
- `discipline_other`

That data does not belong in Git or public example exports.

See also:

- [`../deployment.md`](../deployment.md)
- [`../guides/admin-operations.md`](../guides/admin-operations.md)
