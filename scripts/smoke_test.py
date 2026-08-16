#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys
from tempfile import TemporaryDirectory

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tmis.data.schema import normalize_record
from tmis.data.dataset import TwitterMultimodalDataset
from tmis.data.collator import _trim_field_token_ids, serialize_bridge
from tmis.evaluation import (
    compute_bridge_reference_metrics,
    compute_metrics,
    compute_structure_metrics,
    parse_bridge_text,
)
from tmis.training.losses import (
    bridge_generation_loss,
    reasoning_tag_loss,
    text_evidence_loss,
    visual_evidence_contrastive_loss,
)
from tmis.training.trainer import bridge_selection_score, linear_schedule
from tmis.utils.checkpoint import load_checkpoint, save_checkpoint
from tmis.utils.io import write_json


def main():
    sample = json.loads((ROOT / "tests/fixtures/sample_record.json").read_text(encoding="utf-8"))
    record = normalize_record(sample, 0)
    assert record.is_implicit is True
    assert record.sentiment == "positive"
    text = serialize_bridge(record.reasoning_bridge)
    assert "[GROUND]" in text and "[TRANSITION]" in text and "[IMPLICATION]" in text

    parsed = parse_bridge_text(text)
    assert parsed.structure_valid
    assert parsed.fields["grounded_synthesis"] == record.reasoning_bridge["grounded_synthesis"]
    t5_wrapped = parse_bridge_text(f"<BRIDGE_BOS> {text} </s>")
    assert t5_wrapped.structure_valid
    assert "</s>" not in t5_wrapped.fields["evaluative_implication"]
    structure = compute_structure_metrics([parsed])
    assert structure["valid_rate"] == 1.0

    bridge_metric = compute_bridge_reference_metrics(
        [parsed.fields], [record.reasoning_bridge], compute_bertscore=False
    )
    assert abs(bridge_metric["rouge_l_f1"]["full"] - 1.0) < 1e-9
    stage3_metrics = {"structure": structure, "reference": bridge_metric}
    assert bridge_selection_score(stage3_metrics, "rouge_l_f1_full") == 1.0
    try:
        bridge_selection_score({"structure": structure, "reference": {}}, "rouge_l_f1_full")
    except ValueError:
        pass
    else:
        raise AssertionError("Stage 3 must reject dev data without reference Bridge metrics")

    # Four Stage-5 epochs add a final generated-only adaptation epoch.
    ratios = [linear_schedule(0.25, 1.0, epoch, 4) for epoch in range(4)]
    assert ratios == [0.25, 0.5, 0.75, 1.0]

    # Field-aware truncation keeps three fields represented at the ID-list level.
    trimmed = _trim_field_token_ids([[1, 2, 3, 4], [5, 6, 7], [8, 9]], 6)
    assert sum(map(len, trimmed)) == 6

    # Loss functions should be finite and differentiable.
    te_logits = torch.tensor([[0.0, 1.0, -1.0]], requires_grad=True)
    te_labels = torch.tensor([[0.0, 1.0, -100.0]])
    l_te = text_evidence_loss(te_logits, te_labels)
    assert torch.isfinite(l_te)

    tag_logits = torch.zeros((2, 3), requires_grad=True)
    tag_labels = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    l_tag = reasoning_tag_loss(tag_logits, tag_labels)
    assert torch.isfinite(l_tag)

    v = torch.randn(3, 8, requires_grad=True)
    t = torch.randn(3, 8, requires_grad=True)
    l_ve = visual_evidence_contrastive_loss(v, t, torch.tensor([1, 1, 1], dtype=torch.bool))
    assert torch.isfinite(l_ve)

    # Per-device batch size 1 must still receive real negatives from the queue,
    # while the presence head learns both positive and absent-evidence cases.
    singleton_v = torch.randn(1, 8, requires_grad=True)
    singleton_t = torch.randn(1, 8, requires_grad=True)
    presence_logits = torch.zeros(1, requires_grad=True)
    negative_bank = torch.randn(4, 8)
    singleton_loss = visual_evidence_contrastive_loss(
        singleton_v,
        singleton_t,
        torch.tensor([True]),
        negative_bank=negative_bank,
        presence_logits=presence_logits,
    )
    singleton_loss.backward()
    assert torch.isfinite(singleton_loss) and singleton_v.grad is not None

    # T5 teacher forcing predicts every bridge target token from an equally
    # long, right-shifted decoder input.
    bridge_logits = torch.randn(2, 5, 20, requires_grad=True)
    bridge_ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 0, 0]])
    l_bridge = bridge_generation_loss(
        bridge_logits, bridge_ids, torch.tensor([1, 1], dtype=torch.bool), pad_id=0
    )
    assert torch.isfinite(l_bridge)

    metrics = compute_metrics([0, 1, 2], [0, 1, 2], [True, False, True])
    assert metrics["full"]["macro_f1"] == 1.0

    # Strict boolean schema: strings such as "false" must not silently become True.
    bad = dict(sample)
    bad["reasoning_tags"] = dict(sample["reasoning_tags"])
    bad["reasoning_tags"]["implicit_reasoning_required"] = "false"
    try:
        normalize_record(bad, 1)
    except TypeError:
        pass
    else:
        raise AssertionError("strict boolean validation failed")

    # Checkpoint writes are atomic/restartable and load through weights_only.
    with TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "latest.pt"
        module = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        expected = module.weight.detach().clone()
        save_checkpoint(
            checkpoint,
            module,
            optimizer=optimizer,
            scheduler=scheduler,
            meta={"stage": "stage1_aux", "epoch": 1},
        )
        with torch.no_grad():
            module.weight.zero_()
        meta = load_checkpoint(
            checkpoint,
            module,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        assert meta["epoch"] == 1 and torch.equal(module.weight, expected)
        assert not list(Path(tmp).glob("*.tmp"))

        dataset = object.__new__(TwitterMultimodalDataset)
        dataset.image_dir = Path(tmp).resolve()
        dataset.image_extensions = [".jpg"]
        try:
            dataset._resolve_image("../outside.jpg")
        except ValueError:
            pass
        else:
            raise AssertionError("image path traversal was not rejected")

        # Exported reports must be self-describing and must never copy a model
        # checkpoint into the Git-trackable report directory.
        output_dir = Path(tmp) / "outputs"
        output_dir.mkdir()
        write_json(
            output_dir / "run_state.json",
            {
                "schema_version": 1,
                "run_id": "report-smoke",
                "experiment_name": "report-smoke",
                "status": "completed",
                "started_at_utc": "2026-01-01T00:00:00+00:00",
                "finished_at_utc": "2026-01-01T00:01:00+00:00",
                "git": {"commit": "abc123", "branch": "main", "tracked_worktree_dirty": False},
                "environment": {"cuda_available": False, "gpu_count": 0, "gpus": []},
                "error": None,
            },
        )
        write_json(
            output_dir / "stage1_aux_epoch_1_train.json",
            {
                "stage": "stage1_aux",
                "epoch": 1,
                "routing_gold_mix": 0.0,
                "generated_bridge_ratio": 0.0,
                "trainable_parameters": 10,
                "mixed_precision": "fp32",
                "optimizer": "adafactor",
                "mean_losses": {"text_evidence": 0.5},
            },
        )
        write_json(output_dir / "test_metrics.json", {"full": {"macro_f1": 0.5}})
        torch.save({"model": module.state_dict()}, output_dir / "best_joint.pt")
        reports_root = Path(tmp) / "reports"
        report_dir = reports_root / "twitter2015" / "report-smoke"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/export_training_report.py"),
                "--config",
                str(ROOT / "configs/twitter2015.yaml"),
                "--output-dir",
                str(output_dir),
                "--reports-root",
                str(reports_root),
                "--report-dir",
                str(report_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/validate_training_report.py"),
                "--report-dir",
                str(report_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert (report_dir / "run_manifest.json").is_file()
        assert not list(report_dir.rglob("*.pt"))

    print("SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
