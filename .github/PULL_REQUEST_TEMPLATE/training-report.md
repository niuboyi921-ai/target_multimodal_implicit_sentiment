## Training report

- Run ID:
- Dataset: Twitter-2015 / Twitter-2017
- Report path: `reports/<dataset>/<run-id>/`
- Training commit recorded in manifest:
- Run status: completed / failed / interrupted

## Publisher checklist

- [ ] `python scripts/validate_training_report.py --report-dir <report-path>` passes.
- [ ] No checkpoint, pretrained weight, token, key, or `outputs/` file is included.
- [ ] The report branch contains only files under `reports/`.
- [ ] Full, Implicit, and Non-implicit metrics are present or their absence is explained.
- [ ] Failed/interrupted status is clearly disclosed.

## GPT/Codex analysis request

After opening the pull request, post the following as a PR comment:

```text
@codex Analyze this training run using AGENTS.md. Verify the recorded training commit first, then:

1. Analyze convergence and anomalies in each of the five stages.
2. Compare Full, Implicit, and Non-implicit sentiment results.
3. Assess autoregressive Bridge quality and possible cascading error.
4. Check for class collapse, routing collapse, modality dominance, overfitting, and train/test mismatch.
5. Compare with the most relevant prior completed run if one exists.
6. Separate observed evidence from inference and propose prioritized next experiments.
```
