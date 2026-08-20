#!/usr/bin/env python3
"""Static contract audit; does not download pretrained models."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def class_method_args(tree: ast.Module, class_name: str, method_name: str) -> list[str]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                    return [arg.arg for arg in item.args.args]
    raise AssertionError(f"missing {class_name}.{method_name}")


def main() -> None:
    bridge = parse(ROOT / "src/tmis/models/bridge.py")
    args = class_method_args(bridge, "BridgeEncoderClassifier", "forward")
    assert args == ["self", "input_ids", "attention_mask"], (
        "Bridge classifier API must only consume bridge token IDs and attention mask; "
        f"got {args}"
    )

    encoders_text = (ROOT / "src/tmis/models/encoders.py").read_text(encoding="utf-8")
    assert "T5ForConditionalGeneration" in encoders_text
    assert "CLIPVisionModel" in encoders_text
    assert "AutoModel" not in encoders_text

    bridge_text = (ROOT / "src/tmis/models/bridge.py").read_text(encoding="utf-8")
    assert "encoder_outputs=encoder_outputs" in bridge_text
    assert "t5_model" in bridge_text
    assert "t5_model.generate(" in bridge_text
    assert "use_cache=True" in bridge_text
    assert "memory_type_embeddings" in bridge_text
    assert "nn.Embedding(4, t5_hidden_dim)" in bridge_text
    assert "five multimodal bridge-memory tokens" not in bridge_text
    assert "pretrained_token_embeddings" in bridge_text
    assert "nn.TransformerDecoder" not in bridge_text, (
        "reasoning bridge generation must use the pretrained T5 decoder, "
        "not a separate randomly initialized TransformerDecoder"
    )

    config_texts = {
        name: (ROOT / f"configs/{name}.yaml").read_text(encoding="utf-8")
        for name in ("twitter2015", "twitter2017")
    }
    for name, config_text in config_texts.items():
        assert "text_backbone: google-t5/t5-large" in config_text, name
        assert "vision_backbone: openai/clip-vit-large-patch14" in config_text, name
        assert "text_backbone_revision:" in config_text, name
        assert "vision_backbone_revision:" in config_text, name
        assert "selector_regularization" in config_text, name
        assert "text_evidence:" not in config_text, name
        assert "visual_evidence:" not in config_text, name
        assert "freeze_text_backbone: true" in config_text, name
        assert "freeze_vision_backbone: true" in config_text, name
        assert "freeze_bridge_token_embeddings: true" in config_text, name
        assert "rank: 8" in config_text, name
        assert 'target_modules: ["q", "v"]' in config_text, name
        assert "model: qwen3.7-plus-2026-05-26" in config_text, name
        assert "model: qwen3.8-max" in config_text, name
        assert "random_audit_ratio: 0.10" in config_text, name
        assert "review_cross_modal: true" in config_text, name
        assert "review_order_inconsistent: true" in config_text, name
        assert "review_low_margin: true" in config_text, name

    constants_text = (ROOT / "src/tmis/constants.py").read_text(encoding="utf-8")
    assert 'BRIDGE_EOS_TOKEN = "</s>"' in constants_text

    collator_text = (ROOT / "src/tmis/data/collator.py").read_text(encoding="utf-8")
    assert "target_input_ids" in collator_text
    assert '"token_type_ids"' not in collator_text

    trainer_text = (ROOT / "src/tmis/training/trainer.py").read_text(encoding="utf-8")
    for stage in [
        "stage1_aux",
        "stage2_reasoning_warmup",
        "stage3_bridge",
        "stage4_classifier",
        "stage5_joint",
    ]:
        assert stage in trainer_text, f"missing training stage: {stage}"
    assert "generate_bridge" in trainer_text
    assert "generated_bridge_ratio" in trainer_text
    assert "routing_gold_mix" in trainer_text
    assert "def evaluate_bridge_generation" in trainer_text
    assert 'self.out_dir / "best_bridge.pt"' in trainer_text
    assert "routing_gold_mix=0.0" in trainer_text
    assert "best_meta = load_checkpoint(" in trainer_text
    assert "DistributedDataParallel" in trainer_text
    assert "DistributedSampler" in trainer_text
    assert "torch.amp.GradScaler" in trainer_text
    assert '"stage5_generated_only.pt"' in trainer_text
    assert '"latest.pt"' in trainer_text
    assert "selector_regularization_loss" in trainer_text
    assert "self.model.text_selector" in trainer_text
    assert "self.model.visual_selector" in trainer_text
    assert "text_evidence_loss" not in trainer_text
    assert "visual_evidence_contrastive_loss" not in trainer_text
    assert "_assert_parameter_efficient_contract" in trainer_text
    assert "self.model.bridge_classifier.token_embedding.weight.requires_grad = False" in trainer_text
    assert "frozen Bridge token embeddings became trainable" in trainer_text
    assert '"trainable_bridge_token_embedding_parameters"' in trainer_text
    assert "_set_dpo_trainable" in trainer_text
    assert "trainable_t5_base_parameters" in trainer_text
    assert "trainable_clip_parameters" in trainer_text
    assert "for parameter in self.model.parameters():\n                parameter.requires_grad = True" not in trainer_text
    assert "_collect_stage3_preferences" in trainer_text
    assert "_train_stage3_dpo" in trainer_text
    assert '"uses_gold_sentiment": False' in trainer_text
    assert '"uses_gold_reasoning_tags": False' in trainer_text
    assert '"uses_reference_bridge": False' in trainer_text
    assert "evaluate_stage3_absolute_judge" in trainer_text
    assert "stage3_checkpoint_decision" in trainer_text
    assert "stage5_checkpoint_decision" in trainer_text
    assert "quality_rejected_pairs" in trainer_text

    for name, config_text in config_texts.items():
        assert "checkpoint_selection:" in config_text, name
        assert "min_mean_dimension_score: 3.0" in config_text, name
        assert "quality_gate:" in config_text, name
        assert "implicit_macro_f1_weight: 0.60" in config_text, name
        assert "generated_bridge_ratio_end: 1.0" in config_text, name

    model_text = (ROOT / "src/tmis/models/model.py").read_text(encoding="utf-8")
    assert "if routing_gold_mix > 0 else None" in model_text
    assert "self.text_encoder.backbone" in model_text
    assert "def forward(" in model_text
    assert "self.text_selector" in model_text
    assert "self.visual_selector" in model_text
    assert "h_va[:, 1:, :]" in model_text
    assert "compute_visual_evidence_target" not in model_text
    assert "visual_presence_logits" not in model_text
    assert "self.tag_head(" in model_text
    assert "CrossModalRelationModule" not in model_text
    assert "self.relation" not in model_text
    assert "h_relation" not in model_text
    assert "[h_r, h_text_selected, h_visual_selected, h_a]" in model_text
    assert 'checkpoint_state_format = "parameter_efficient_v1"' in model_text
    assert "checkpoint_state_dict" in model_text
    assert "bridge_classifier.token_embedding.weight.requires_grad_(False)" in model_text

    encoders_text = (ROOT / "src/tmis/models/encoders.py").read_text(encoding="utf-8")
    assert "inject_lora(" in encoders_text
    assert "parameter.requires_grad = False" in encoders_text
    lora_text = (ROOT / "src/tmis/models/lora.py").read_text(encoding="utf-8")
    assert "class LoRALinear" in lora_text
    assert "nn.init.zeros_(self.lora_b.weight)" in lora_text
    assert "def inject_lora" in lora_text

    loss_text = (ROOT / "src/tmis/training/losses.py").read_text(encoding="utf-8")
    for name in [
        "selector_regularization_loss",
        "reasoning_tag_loss",
        "bridge_generation_loss",
        "sentiment_loss",
        "sequence_log_probs",
        "dpo_preference_loss",
    ]:
        assert f"def {name}" in loss_text
    assert "def text_evidence_loss" not in loss_text
    assert "def visual_evidence_contrastive_loss" not in loss_text

    ai_feedback_text = (
        ROOT / "src/tmis/training/ai_feedback.py"
    ).read_text(encoding="utf-8")
    assert "load_bailian_credentials" in ai_feedback_text
    assert "bailian_credentials_local" in ai_feedback_text
    assert "os.getenv" not in ai_feedback_text
    assert "PAIRWISE_JUDGE_SYSTEM_PROMPT" in ai_feedback_text
    assert "ABSOLUTE_JUDGE_SYSTEM_PROMPT" in ai_feedback_text
    assert "class AbsoluteBailianJudge" in ai_feedback_text
    assert "def passes_quality_gate" in ai_feedback_text
    assert "candidate_A" in ai_feedback_text and "candidate_B" in ai_feedback_text

    gitignore_text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "/src/tmis/bailian_credentials_local.py" in gitignore_text

    selectors_text = (ROOT / "src/tmis/models/selectors.py").read_text(encoding="utf-8")
    assert "class TargetAwareTextSelector" in selectors_text
    assert "class TargetAwareVisualSelector" in selectors_text

    reasoning = parse(ROOT / "src/tmis/models/reasoning.py")
    tag_args = class_method_args(reasoning, "ReasoningTagHead", "forward")
    assert tag_args == [
        "self",
        "h_text_selected",
        "h_visual_selected",
    ], f"ReasoningTagHead must consume only both selector outputs; got {tag_args}"
    reasoner_args = class_method_args(reasoning, "MultiPathReasoner", "forward")
    assert reasoner_args == [
        "self",
        "h_text_selected",
        "h_visual_selected",
        "h_text_global",
        "h_a",
    ], (
        "MultiPathReasoner must keep Direct/Implicit text-only and let the Cross "
        f"path compare selected text/visual features directly; got {reasoner_args}"
    )
    reasoning_text = (ROOT / "src/tmis/models/reasoning.py").read_text(encoding="utf-8")
    assert "self.direct(h_text_selected, h_a)" in reasoning_text
    assert "self.implicit(h_text_selected, h_text_global, h_a)" in reasoning_text
    assert "torch.abs(h_text_selected - h_visual_selected)" in reasoning_text
    assert "h_text_selected * h_visual_selected" in reasoning_text
    assert "class CrossModalRelationModule" not in reasoning_text

    schema_text = (ROOT / "src/tmis/data/schema.py").read_text(encoding="utf-8")
    assert "legacy artificial evidence fields are not supported" in schema_text
    fixture_text = (ROOT / "tests/fixtures/sample_record.json").read_text(encoding="utf-8")
    assert '"text_evidence"' not in fixture_text
    assert '"visual_evidence"' not in fixture_text

    checkpoint_text = (ROOT / "src/tmis/utils/checkpoint.py").read_text(encoding="utf-8")
    assert "os.replace" in checkpoint_text
    assert "weights_only=True" in checkpoint_text
    assert "model_state_format" in checkpoint_text

    dataset_text = (ROOT / "src/tmis/data/dataset.py").read_text(encoding="utf-8")
    assert "candidate.relative_to(self.image_dir)" in dataset_text

    print("ARCHITECTURE_AUDIT_OK")


if __name__ == "__main__":
    main()
