# Backend Bootstrap Pack

This folder holds the zip used for startup bootstrap on ephemeral hosting.

- Default pack path: `services/api/bootstrap/final_expert_pack.zip`
- Enable with: `BOOTSTRAP_PACK_ON_STARTUP=true`
- The backend imports this pack at startup if the active campaign is missing or has missing assets.
- Current bootstrap manifest mode: `two_block` (`block_a_target_count=10`, `block_b_target_count=12`).
- Three-part mode is supported via `block_c_items` in `manifest_public.json`.
  - Recommended setup for a three-part protocol: `block_c_items=16`.
  - In three-part mode, `block_b_target_count` should be total pairwise rows (`Block B + Block C`).
- Image layout inside the zip is block-grouped:
  - `images/block_a/<sample_id>/...`
  - `images/block_b/<sample_id>/...`
  - `images/block_c/<sample_id>/...`

If you need to normalize an older zip layout, run:

```bash
python scripts/utilities/repack_expert_zip_by_block.py --zip-path services/api/bootstrap/final_expert_pack.zip
```

Keep this pack aligned with the currently active expert-validation campaign.
