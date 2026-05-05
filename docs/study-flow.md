# Website Study Flow

The website is a guided expert-evaluation system, not a generic survey builder.

## Participant flow

`WELCOME -> CONSENT -> PROFILE -> BLOCK_A -> BLOCK_A_FEEDBACK -> BLOCK_B -> BLOCK_B_FEEDBACK -> COMPLETE`

Rules:

- new participants pass through `PROFILE`
- already in-progress participants are not forced backwards
- Block B unlocks only after Block A item completion and required Block A feedback
- submitted item responses remain locked

## What is collected

Optional profile fields:

- `name`
- `institution`
- `discipline`
- `discipline_other`

Block A:

- authenticity likelihood
- archaeological plausibility
- confidence
- required per-item comment (`trimmed non-empty`)
- `response_time_ms`

Block B:

- choice (`A|B|Tie|Unsure`)
- confidence
- required per-item comment (`trimmed non-empty`)
- `response_time_ms`

Wrap-up feedback:

- one required comment after Block A
- one required comment after Block B

## Attention checks

- Block A: 2 repeated-image checks
- Block B: 1 repeated-pair check

## Privacy boundary

Exports remain keyed by anonymized participant IDs, but optional profile fields still count as personal data when supplied.

See also:

- [`architecture.md`](architecture.md)
- [`api-contracts.md`](api-contracts.md)
- [`admin-operations.md`](admin-operations.md)
