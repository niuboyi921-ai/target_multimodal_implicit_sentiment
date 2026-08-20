#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import subprocess
import sys
from tempfile import TemporaryDirectory

from PIL import Image
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
    dpo_preference_loss,
    reasoning_tag_loss,
    sequence_log_probs,
    selector_regularization_loss,
    sentiment_loss,
)
from tmis.training.ai_feedback import (
    PairwiseBailianJudge,
    passes_quality_gate,
    score_margin,
    stable_audit_pick,
    swap_result_to_original,
    validate_absolute_result,
    validate_pairwise_result,
)
from tmis.training.trainer import (
    bridge_selection_score,
    effective_number_class_weights,
    linear_schedule,
    stage3_checkpoint_decision,
    stage5_checkpoint_decision,
)
from tmis.models.lora import LoRALinear, inject_lora, is_lora_parameter
from tmis.utils.checkpoint import load_checkpoint, save_checkpoint
from tmis.utils.io import write_json


def _load_lightweight_module(name: str, path: Path):
    """Load torch-only model modules without importing optional Transformers."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_reasoning = _load_lightweight_module(
    "tmis_smoke_reasoning", ROOT / "src/tmis/models/reasoning.py"
)
_selectors = _load_lightweight_module(
    "tmis_smoke_selectors", ROOT / "src/tmis/models/selectors.py"
)
ReasoningTagHead = _reasoning.ReasoningTagHead
MultiPathReasoner = _reasoning.MultiPathReasoner
TargetAwareTextSelector = _selectors.TargetAwareTextSelector
TargetAwareVisualSelector = _selectors.TargetAwareVisualSelector


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

    absolute = validate_absolute_result(
        {
            "scores": {
                "faithfulness": 4,
                "reasoning_coherence": 4,
                "target_consistency": 5,
            },
            "critical_error": False,
            "rationale": "grounded and target-consistent",
        }
    )
    assert passes_quality_gate(
        absolute["scores"],
        {
            "min_faithfulness": 3,
            "min_reasoning_coherence": 3,
            "min_target_consistency": 3,
            "min_total_score": 10,
        },
    )
    assert not passes_quality_gate(
        {"faithfulness": 2, "reasoning_coherence": 5, "target_consistency": 5},
        {"min_faithfulness": 3, "min_total_score": 10},
    )
    stage3_absolute_metrics = {
        "structure": {"valid_rate": 0.98},
        "reference": {"rouge_l_f1": {"full": 0.42}},
        "absolute_judge": {
            "sample_size": 8,
            "dimension_means": absolute["scores"],
            "critical_error_rate": 0.0,
        },
    }
    stage3_decision = stage3_checkpoint_decision(
        stage3_absolute_metrics,
        {
            "min_structure_valid_rate": 0.90,
            "min_mean_dimension_score": 3.0,
            "max_critical_error_rate": 0.20,
        },
    )
    assert stage3_decision["eligible"] is True
    assert stage3_decision["rouge_l_f1_tiebreak"] == 0.42

    # Four Stage-5 epochs add a final generated-only adaptation epoch.
    ratios = [linear_schedule(0.25, 1.0, epoch, 4) for epoch in range(4)]
    assert ratios == [0.25, 0.5, 0.75, 1.0]

    # Native LoRA must preserve the frozen layer's initial output and expose
    # trainable low-rank matrices without modifying the base weights.
    class TinyAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q = torch.nn.Linear(8, 8, bias=False)
            self.k = torch.nn.Linear(8, 8, bias=False)
            self.v = torch.nn.Linear(8, 8, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.q(x) + self.k(x) + self.v(x)

    tiny = TinyAttention()
    lora_input = torch.randn(2, 8)
    baseline = tiny(lora_input).detach()
    injected = inject_lora(
        tiny,
        target_modules=["q", "v"],
        rank=2,
        alpha=4,
        dropout=0.0,
    )
    assert injected == ("q", "v")
    assert isinstance(tiny.q, LoRALinear) and isinstance(tiny.v, LoRALinear)
    assert torch.allclose(tiny(lora_input), baseline, atol=1e-6)
    assert tiny.q.base_layer.weight.requires_grad is False
    assert tiny.v.base_layer.weight.requires_grad is False
    assert tiny.k.weight.requires_grad is True  # stage trainer performs global freezing
    assert is_lora_parameter("decoder.q.lora_a.weight")
    tiny(lora_input).sum().backward()
    assert tiny.q.lora_b.weight.grad is not None
    assert tiny.q.base_layer.weight.grad is None

    # Field-aware truncation keeps three fields represented at the ID-list level.
    trimmed = _trim_field_token_ids([[1, 2, 3, 4], [5, 6, 7], [8, 9]], 6)
    assert sum(map(len, trimmed)) == 6

    # Loss functions should be finite and differentiable.
    text_logits = torch.zeros((2, 4), requires_grad=True)
    text_weights = torch.sigmoid(text_logits)
    text_mask = torch.tensor(
        [[True, True, True, False], [True, True, False, False]]
    )
    visual_logits = torch.zeros((2, 5), requires_grad=True)
    visual_weights = torch.softmax(visual_logits, dim=-1)
    l_selector = selector_regularization_loss(
        text_weights,
        text_mask,
        visual_weights,
    )
    assert torch.isfinite(l_selector)
    l_selector.backward()
    assert text_logits.grad is not None and visual_logits.grad is not None

    # Reasoning-tag supervision must reach both latent modality selectors.
    tag_labels = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    hidden = 8
    text_selector = TargetAwareTextSelector(hidden, dropout=0.0)
    visual_selector = TargetAwareVisualSelector(hidden, dropout=0.0)
    tag_head = ReasoningTagHead(hidden, dropout=0.0)
    h_text = torch.randn(2, 4, hidden, requires_grad=True)
    h_visual = torch.randn(2, 5, hidden, requires_grad=True)
    h_fused = torch.randn(2, hidden, requires_grad=True)
    selector_mask = torch.tensor(
        [[True, True, True, False], [True, True, False, False]]
    )
    _, learned_text_weights, h_text_selected = text_selector(
        h_text, h_fused, selector_mask
    )
    h_visual_selected, learned_visual_weights = visual_selector(h_visual, h_fused)
    learned_tag_logits, _ = tag_head(
        h_text_selected, h_visual_selected
    )
    selector_guided_loss = reasoning_tag_loss(learned_tag_logits, tag_labels)
    selector_guided_loss = selector_guided_loss + 0.1 * selector_regularization_loss(
        learned_text_weights,
        selector_mask,
        learned_visual_weights,
    )
    selector_guided_loss.backward()
    assert h_text.grad is not None and h_visual.grad is not None
    assert any(parameter.grad is not None for parameter in text_selector.parameters())
    assert any(parameter.grad is not None for parameter in visual_selector.parameters())

    # Path modality contract: Direct is selected-text + target; Implicit adds
    # only global text context; Cross alone receives selected visual features
    # and computes deterministic product/difference comparisons internally.
    # The public API has neither a fused h_f nor a learned relation argument.
    reasoner = MultiPathReasoner(hidden, dropout=0.0)
    path_text = h_text_selected.detach().clone().requires_grad_(True)
    path_visual = h_visual_selected.detach().clone().requires_grad_(True)
    h_text_global = torch.randn(2, hidden, requires_grad=True)
    h_target = torch.randn(2, hidden, requires_grad=True)
    paths = reasoner(
        path_text,
        path_visual,
        h_text_global,
        h_target,
    )
    assert len(paths) == 3
    assert all(path.shape == (2, hidden) for path in paths)
    path_inputs = (path_text, path_visual, h_text_global, h_target)
    direct_grads = torch.autograd.grad(
        paths[0].sum(), path_inputs, retain_graph=True, allow_unused=True
    )
    assert direct_grads[0] is not None and direct_grads[3] is not None
    assert direct_grads[1] is None and direct_grads[2] is None
    implicit_grads = torch.autograd.grad(
        paths[1].sum(), path_inputs, retain_graph=True, allow_unused=True
    )
    assert implicit_grads[0] is not None and implicit_grads[2] is not None
    assert implicit_grads[3] is not None
    assert implicit_grads[1] is None
    cross_grads = torch.autograd.grad(
        paths[2].sum(), path_inputs, allow_unused=True
    )
    assert cross_grads[0] is not None and cross_grads[1] is not None
    assert cross_grads[3] is not None
    assert cross_grads[2] is None

    tag_logits = torch.zeros((2, 3), requires_grad=True)
    l_tag = reasoning_tag_loss(tag_logits, tag_labels)
    assert torch.isfinite(l_tag)

    # Effective-number weights keep their expected training-distribution scale
    # at one and must remain active for the per-device batch size of one.
    # SENTIMENT_TO_ID order: positive, neutral, negative.
    sentiment_counts = torch.tensor([842.0, 1855.0, 319.0])
    sentiment_weights = effective_number_class_weights(sentiment_counts, beta=0.999)
    class_probability = sentiment_counts / sentiment_counts.sum()
    assert torch.allclose(
        (class_probability * sentiment_weights).sum(),
        torch.tensor(1.0),
        atol=1e-6,
    )
    assert sentiment_weights[2] > sentiment_weights[0] > sentiment_weights[1]
    singleton_logits = torch.zeros((1, 3), requires_grad=True)
    singleton_label = torch.tensor([2])
    singleton_sentiment_loss = sentiment_loss(
        singleton_logits,
        singleton_label,
        class_weight=sentiment_weights,
    )
    unweighted_singleton_loss = torch.nn.functional.cross_entropy(
        singleton_logits,
        singleton_label,
    )
    assert torch.allclose(
        singleton_sentiment_loss,
        unweighted_singleton_loss * sentiment_weights[2],
    )
    singleton_sentiment_loss.backward()
    assert singleton_logits.grad is not None

    # T5 teacher forcing predicts every bridge target token from an equally
    # long, right-shifted decoder input.
    bridge_logits = torch.randn(2, 5, 20, requires_grad=True)
    bridge_ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 0, 0]])
    l_bridge = bridge_generation_loss(
        bridge_logits, bridge_ids, torch.tensor([1, 1], dtype=torch.bool), pad_id=0
    )
    assert torch.isfinite(l_bridge)

    # Pairwise Bridge preference learning remains differentiable only through
    # local T5 log-probabilities; remote Judge decisions are cached data.
    preference_logits = torch.randn(2, 5, 20, requires_grad=True)
    preference_mask = bridge_ids.ne(0).long()
    candidate_logps = sequence_log_probs(
        preference_logits,
        bridge_ids,
        preference_mask,
        pad_id=0,
        length_normalize=True,
    )
    l_dpo = dpo_preference_loss(
        candidate_logps[:1],
        candidate_logps[1:],
        torch.tensor([-2.0]),
        torch.tensor([-2.1]),
        beta=0.1,
    )
    assert torch.isfinite(l_dpo)
    l_dpo.backward()
    assert preference_logits.grad is not None

    judge_result = validate_pairwise_result(
        {
            "winner": "A",
            "scores": {
                "A": {
                    "faithfulness": 5,
                    "reasoning_coherence": 4,
                    "target_consistency": 5,
                },
                "B": {
                    "faithfulness": 2,
                    "reasoning_coherence": 3,
                    "target_consistency": 2,
                },
            },
            "critical_error": {"A": False, "B": True},
            "rationale": "A is grounded and target-consistent.",
        }
    )
    assert judge_result["winner"] == "A"
    assert score_margin(judge_result) > 0
    swapped = swap_result_to_original(judge_result)
    assert swapped["winner"] == "B"
    assert stable_audit_pick("same-pair", 0.10) == stable_audit_pick(
        "same-pair", 0.10
    )

    # The review tier is conditional and the reversed primary verdict is
    # mapped back to the original A/B order before checking inconsistency.
    mock_judge = PairwiseBailianJudge.__new__(PairwiseBailianJudge)
    mock_judge.cfg = {
        "primary_judge": {"model": "primary"},
        "review_judge": {"model": "review"},
        "review_rules": {
            "random_audit_ratio": 0.0,
            "review_cross_modal": True,
            "review_order_inconsistent": True,
            "review_low_margin": False,
        },
    }
    mock_judge.image_max_side = 32
    mock_responses = [judge_result, judge_result, judge_result]
    mock_judge._call = lambda model_cfg, **kwargs: mock_responses.pop(0)
    mock_decision = mock_judge.compare(
        text="A target-level post",
        target="target",
        image=Image.new("RGB", (4, 4), color="white"),
        candidate_a="[GROUND] a [TRANSITION] b [IMPLICATION] c",
        candidate_b="[GROUND] d [TRANSITION] e [IMPLICATION] f",
        cross_modal=True,
        audit_material="mock",
    )
    assert mock_decision.winner == "A"
    assert mock_decision.review is not None
    assert "cross_modal" in mock_decision.review_reasons
    assert "order_inconsistent" in mock_decision.review_reasons

    metrics = compute_metrics([0, 1, 2], [0, 1, 2], [True, False, True])
    assert metrics["full"]["macro_f1"] == 1.0
    assert metrics["full"]["prediction_counts"] == {
        "positive": 1,
        "neutral": 1,
        "negative": 1,
    }
    metrics["bridge_structure"] = {"valid_rate": 1.0}
    stage5_decision = stage5_checkpoint_decision(
        metrics,
        {
            "full_macro_f1_weight": 0.4,
            "implicit_macro_f1_weight": 0.6,
            "min_predictions_per_class": 1,
            "min_negative_recall": 0.01,
            "max_implicit_macro_f1_drop": 0.02,
            "min_bridge_structure_valid_rate": 0.9,
        },
        best_implicit_macro_f1=0.9,
    )
    assert stage5_decision["eligible"] is True

    # Strict boolean schema: strings such as "false" must not silently become True.
    bad = dict(sample)
    bad["reasoning_tags"] = dict(sample["reasoning_tags"])
    bad["reasoning_tags"]["implicit_sentiment_present"] = "false"
    try:
        normalize_record(bad, 1)
    except TypeError:
        pass
    else:
        raise AssertionError("strict boolean validation failed")

    legacy_evidence = dict(sample)
    legacy_evidence["text_evidence"] = ["Chuck Bass is everything"]
    try:
        normalize_record(legacy_evidence, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("legacy artificial-evidence field was not rejected")

    # The strict implicit-polarity definition excludes neutral examples.
    neutral_implicit = dict(sample)
    neutral_implicit["sentiment"] = "neutral"
    neutral_implicit["reasoning_tags"] = dict(sample["reasoning_tags"])
    try:
        normalize_record(neutral_implicit, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("neutral implicit sentiment validation failed")

    conflicting_tags = dict(sample)
    conflicting_tags["reasoning_tags"] = dict(sample["reasoning_tags"])
    conflicting_tags["reasoning_tags"]["explicit_cue_present"] = True
    cooccurring = normalize_record(conflicting_tags, 4)
    assert cooccurring.reasoning_tags["explicit_cue_present"] is True
    assert cooccurring.reasoning_tags["implicit_sentiment_present"] is True

    polar_untagged = dict(sample)
    polar_untagged["reasoning_tags"] = dict(sample["reasoning_tags"])
    polar_untagged["reasoning_tags"]["implicit_sentiment_present"] = False
    try:
        normalize_record(polar_untagged, 5)
    except ValueError:
        pass
    else:
        raise AssertionError("polar explicit/implicit at-least-one validation failed")

    # Checkpoint writes are atomic/restartable and load through weights_only.
    with TemporaryDirectory() as tmp:
        # Keep the report smoke test independent from real training data.
        # GitHub Actions intentionally has no data/ directory.
        smoke_config = Path(tmp) / "configs/twitter2015.yaml"
        smoke_config.parent.mkdir(parents=True, exist_ok=True)
        smoke_config.write_text(
            (ROOT / "configs/twitter2015.yaml").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        smoke_data_dir = Path(tmp) / "data/twitter2015"
        for split in ("train", "dev", "test"):
            write_json(smoke_data_dir / f"{split}.json", [sample])
        smoke_image_dir = smoke_data_dir / "images"
        smoke_image_dir.mkdir(parents=True, exist_ok=True)
        (smoke_image_dir / "1860693.jpg").write_bytes(b"smoke-image")

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

        # Parameter-efficient checkpoints must omit reconstructed frozen state,
        # restore the adapter exactly, and reject ambiguous legacy checkpoints.
        class TinySelectiveModel(torch.nn.Module):
            checkpoint_state_format = "parameter_efficient_v1"

            def __init__(self) -> None:
                super().__init__()
                self.frozen_base = torch.nn.Linear(2, 2, bias=False)
                self.adapter = torch.nn.Linear(2, 2, bias=False)

            def checkpoint_state_dict(self):
                return {"adapter.weight": self.adapter.weight.detach().clone()}

            def load_checkpoint_state_dict(self, state):
                incompatible = self.load_state_dict(state, strict=False)
                assert incompatible.missing_keys == ["frozen_base.weight"]
                assert incompatible.unexpected_keys == []

        selective = TinySelectiveModel()
        selective_path = Path(tmp) / "parameter_efficient.pt"
        expected_adapter = selective.adapter.weight.detach().clone()
        save_checkpoint(selective_path, selective, meta={"format_test": True})
        raw_selective = torch.load(selective_path, map_location="cpu", weights_only=True)
        assert raw_selective["model_state_format"] == "parameter_efficient_v1"
        assert set(raw_selective["model"]) == {"adapter.weight"}
        with torch.no_grad():
            selective.adapter.weight.zero_()
        selective_meta = load_checkpoint(selective_path, selective)
        assert selective_meta["format_test"] is True
        assert torch.equal(selective.adapter.weight, expected_adapter)

        legacy_path = Path(tmp) / "legacy.pt"
        torch.save({"model": selective.state_dict()}, legacy_path)
        try:
            load_checkpoint(legacy_path, selective)
        except ValueError as exc:
            assert "start a new run" in str(exc)
        else:
            raise AssertionError("legacy checkpoint was accepted by the LoRA architecture")

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
                "mean_losses": {"selector_regularization": 0.05},
            },
        )
        write_json(output_dir / "test_metrics.json", {"full": {"macro_f1": 0.5}})
        tagged_metric = {
            "checkpoint": {"name": "best_joint.pt"},
            "sentiment": {
                subset: {
                    "n": 1,
                    "accuracy": 0.5,
                    "macro_f1": 0.5,
                    "macro_recall": 0.5,
                }
                for subset in ("full", "implicit", "non_implicit")
            },
            "bridge_structure": {"valid_rate": 1.0},
        }
        write_json(output_dir / "test_metrics_best_joint.json", tagged_metric)
        generated_metric = dict(tagged_metric)
        generated_metric["checkpoint"] = {"name": "stage5_generated_only.pt"}
        write_json(output_dir / "test_metrics_generated_only.json", generated_metric)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/compare_checkpoint_evaluations.py"),
                "--output-dir",
                str(output_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert (output_dir / "test_checkpoint_comparison.json").is_file()
        write_json(
            output_dir / "stage3_ai_feedback_epoch_1_preferences.json",
            {
                "summary": {
                    "preference_pairs": 2,
                    "quality_rejected_pairs": 1,
                    "reviewed_pairs": 1,
                    "judge_usage": {"api_calls": 3, "cache_hits": 1},
                },
                "preferences": [],
            },
        )
        torch.save({"model": module.state_dict()}, output_dir / "best_joint.pt")
        reports_root = Path(tmp) / "reports"
        report_dir = reports_root / "twitter2015" / "report-smoke"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/export_training_report.py"),
                "--config",
                str(smoke_config),
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
        report_manifest = json.loads(
            (report_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert report_manifest["training_judge"]["used_during_training"] is True
        assert (
            report_manifest["training_judge"]["remote_api_called_during_training"]
            is True
        )
        assert not list(report_dir.rglob("*.pt"))

    print("SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
