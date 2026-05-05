# Troubleshooting

## Common failures

### Captioning OOM in `S07`

Check:

```bash
cat outputs/S07_caption_generation/caption_generation_report.json
```

### Mask realism failure in `S11`

Check:

```bash
cat outputs/S11_feature_engineering/mask_realism_guardrails.json
```

### Evaluation hard-stop in `S15`

Check:

```powershell
dir data\intermediate\10_models
```

### Divergent git history on GCP

Fix:

```bash
git fetch origin
git checkout main
git reset --hard origin/main
```

### GPU occupancy

Check:

```bash
nvidia-smi
```
