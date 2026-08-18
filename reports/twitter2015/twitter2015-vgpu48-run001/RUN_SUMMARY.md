# Training run: twitter2015-vgpu48-run001

- Status: `completed`
- Experiment: `twitter2015_evidence_multipath`
- Dataset: `twitter2015`
- Training commit: `5905072e661e2944afd108431079ca35dabcc583`
- Training branch: `main`
- Tracked worktree dirty: `False`
- Started: `2026-08-18T01:34:49.049690+00:00`
- Finished: `2026-08-18T05:22:37.415690+00:00`

## Hardware

- CUDA available: `True`
- CUDA version: `12.8`
- GPU count: `1`
- GPU 0: `NVIDIA GeForce RTX 4090` (47.4 GiB)

## Epoch summaries

| stage | epoch | loss_bridge | loss_reasoning_tags | loss_sentiment | loss_text_evidence | loss_visual_evidence |
| --- | --- | --- | --- | --- | --- | --- |
| stage1_aux | 1 |  | 0.922351 |  | 0.279722 | 4.5967 |
| stage1_aux | 2 |  | 0.921715 |  | 0.272255 | 4.57615 |
| stage1_aux | 3 |  | 0.921291 |  | 0.268721 | 4.58549 |
| stage2_reasoning_warmup | 1 | 13.8307 | 0.921837 |  |  |  |
| stage2_reasoning_warmup | 2 | 13.2717 | 0.921458 |  |  |  |
| stage3_bridge | 1 | 12.9193 | 0.921472 |  | 0.268432 | 4.46472 |
| stage3_bridge | 2 | 12.459 | 0.92138 |  | 0.268092 | 4.51775 |
| stage3_bridge | 3 | 12.116 | 0.921004 |  | 0.267943 | 4.56523 |
| stage3_bridge | 4 | 11.9872 | 0.921431 |  | 0.267575 | 4.59396 |
| stage4_classifier | 1 |  |  | 0.796784 |  |  |
| stage4_classifier | 2 |  |  | 0.474243 |  |  |
| stage4_classifier | 3 |  |  | 0.382691 |  |  |
| stage5_joint | 1 | 11.9664 | 0.921763 | 0.629468 | 0.267658 | 4.58854 |
| stage5_joint | 2 | 11.9017 | 0.921645 | 0.729751 | 0.267617 | 4.58761 |
| stage5_joint | 3 | 11.8422 | 0.921178 | 0.825819 | 0.268017 | 4.59016 |
| stage5_joint | 4 | 11.8501 | 0.921414 | 0.941437 | 0.267776 | 4.59567 |

## Analysis contract

Treat JSON metrics as observed results and architectural explanations as inferences from the referenced code commit. Do not infer checkpoint quality from file size.
